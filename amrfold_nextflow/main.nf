nextflow.enable.dsl=2

params.input                 = null
params.input_fasta           = null
params.models_dir            = 'models'
params.model_dir             = null
params.outdir                = 'results'

params.chunk_size            = 256
params.map_size_gb           = 8

params.device                = 'auto'
params.half_precision        = true
params.prostt5_dir           = null
params.local_files_only      = false

params.extract_batch_size    = 4
params.extract_token_budget  = 16384
params.extract_commit_every  = 100

params.infer_batch_size      = 4
params.num_workers           = 0

params.bin_threshold         = 0.39917078614234924
params.gate_classes          = 'multidrug,others,sulfonamide,rifamycin,quinolone'
params.gate_tau              = 0.7108869316825098
params.gate_delta            = 0.10345399260721301

params.dropout               = 0.1038155492514013
params.max_len               = 1024
params.class_loss_type       = 'bce'

params.publish_intermediates = false

params.slurm_partition       = null
params.slurm_gpu_partition   = null
params.slurm_account         = null
params.gpus                  = 1
params.gpus_per_task         = 1
params.use_gpu_lock          = false


process SPLIT_FASTA {
    tag { input_fasta.simpleName }
    label 'light'
    publishDir "${params.outdir}/split", mode: 'copy', enabled: params.publish_intermediates

    input:
    path input_fasta

    output:
    path 'shards/*.fa', emit: shards
    path 'manifest.tsv', emit: manifest

    script:
    """
    mkdir -p shards
    python ${projectDir}/bin/amrfold_split_fasta.py \
      --input ${input_fasta} \
      --outdir shards \
      --manifest manifest.tsv \
      --seqs-per-shard ${params.chunk_size}
    """
}


process EXTRACT_FEATURES {
    tag { shard_fasta.simpleName }
    label 'extract'
    publishDir "${params.outdir}/features_lmdb", mode: 'copy', enabled: params.publish_intermediates

    input:
    path shard_fasta

    output:
    tuple path(shard_fasta), path("${shard_fasta.simpleName}.lmdb")

    script:
    def modelsBase = params.models_dir ?: params.model_dir
    def prostDir = params.prostt5_dir ?: (file("${modelsBase}/ProstT5").exists() ? "${modelsBase}/ProstT5" : null)
    def prost = prostDir ? "--prostt5_dir ${prostDir}" : ""
    def localOnly = params.local_files_only ? '--local_files_only true' : '--local_files_only false'
    def gpuWrap = (params.use_gpu_lock && params.device == 'cuda') ? "bash ${projectDir}/bin/acquire_gpu.sh " : ""

    """
    ${gpuWrap}python ${projectDir}/bin/extract_features_fasta.py \
      --input_fasta ${shard_fasta} \
      --lmdb_dir ${shard_fasta.simpleName}.lmdb \
      --map_size_gb ${params.map_size_gb} \
      --device ${params.device} \
      --half_precision ${params.half_precision} \
      --batch_size ${params.extract_batch_size} \
      --token_budget ${params.extract_token_budget} \
      --commit_every ${params.extract_commit_every} \
      ${localOnly} \
      ${prost}
    """
}


process SCORE_LMDB {
    tag { shard_fasta.simpleName }
    label 'score'
    publishDir "${params.outdir}/shards", mode: 'copy', enabled: true, pattern: '*.predictions.tsv'

    input:
    tuple path(shard_fasta), path(features_lmdb)
    path checkpoints

    output:
    path '*.predictions.tsv', emit: shard_predictions

    script:
    def outname = shard_fasta.simpleName + '.predictions.tsv'
    def ckpts = checkpoints.collect { it.getName() }.join(' ')
    def gpuWrap = (params.use_gpu_lock && params.device == 'cuda') ? "bash ${projectDir}/bin/acquire_gpu.sh " : ""

    """
    ${gpuWrap}python ${projectDir}/bin/score_lmdb_ensemble.py \
      --input_fasta ${shard_fasta} \
      --features_lmdb ${features_lmdb} \
      --output ${outname} \
      --device ${params.device} \
      --batch_size ${params.infer_batch_size} \
      --num_workers ${params.num_workers} \
      --bin_threshold ${params.bin_threshold} \
      --gate_classes ${params.gate_classes} \
      --gate_tau ${params.gate_tau} \
      --gate_delta ${params.gate_delta} \
      --dropout ${params.dropout} \
      --max_len ${params.max_len} \
      --class_loss_type ${params.class_loss_type} \
      --checkpoints ${ckpts}
    """
}


process MERGE_PREDICTIONS {
    tag 'merge'
    label 'light'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path shard_tables

    output:
    path 'amrfold_predictions.tsv'

    script:
    def inputs = shard_tables.collect { it.getName() }.join(' ')

    """
    python ${projectDir}/bin/amrfold_merge_predictions.py \
      --output amrfold_predictions.tsv ${inputs}
    """
}


workflow {
    def input_path  = params.input ?: params.input_fasta
    def models_path = params.models_dir ?: params.model_dir

    if( !input_path ) {
        error "Please provide --input or --input_fasta <proteins.faa[.gz]>"
    }

    if( !models_path ) {
        error "Please provide --models_dir or --model_dir <checkpoint directory>"
    }

    def resolvedProstDir = params.prostt5_dir ?: (
        file("${models_path}/ProstT5").exists() ? "${models_path}/ProstT5" : null
    )

    log.info "AMR-Fold input: ${input_path}"
    log.info "AMR-Fold models_dir: ${models_path}"
    log.info "Inference batch size: ${params.infer_batch_size}"
    log.info "Extraction batch size: ${params.extract_batch_size}"
    log.info "Extraction token budget: ${params.extract_token_budget}"
    log.info "Class loss type: ${params.class_loss_type}"
    log.info "Max length: ${params.max_len}"
    log.info "Dropout: ${params.dropout}"
    log.info "Binary threshold: ${params.bin_threshold}"
    log.info "Rare-class gate tau: ${params.gate_tau}"
    log.info "Rare-class gate delta: ${params.gate_delta}"

    if( resolvedProstDir ) {
        log.info "Using local ProstT5 directory: ${resolvedProstDir}"
    }
    else {
        log.info "Using ProstT5 from model name: Rostlab/ProstT5"
    }

    ch_input = Channel.fromPath(input_path, checkIfExists: true)

    ch_ckpts = Channel
        .fromPath("${models_path}/best_checkpoint_seed_*.pt", checkIfExists: true)
        .ifEmpty { error "No checkpoints found under ${models_path}" }
        .collect()

    split_res = SPLIT_FASTA(ch_input)
    feat_res  = EXTRACT_FEATURES(split_res.shards.flatten())
    pred_res  = SCORE_LMDB(feat_res, ch_ckpts)

    MERGE_PREDICTIONS(pred_res.shard_predictions.collect())
}
