nextflow.enable.dsl=2

params.input                   = null
params.input_fasta             = null
params.models_dir              = 'models'
params.model_dir               = null
params.outdir                  = 'results'

params.chunk_size              = 256
params.map_size_gb             = 32

params.device                  = 'cuda'
params.score_device            = 'cuda'
params.half_precision          = true
params.prostt5_dir             = null
params.local_files_only        = false

params.extract_groups          = 2
params.score_groups            = 2
params.score_after_all_extract = true

params.extract_batch_size      = 8
params.extract_token_budget    = 4096
params.extract_commit_every    = 100

params.infer_batch_size        = 16
params.num_workers             = 0

params.bin_threshold           = 0.39917078614234924
params.gate_classes            = 'multidrug,others,sulfonamide,rifamycin,quinolone'
params.gate_tau                = 0.7108869316825098
params.gate_delta              = 0.10345399260721301

params.python_cpu              = 'python'
params.python_gpu              = 'python'
params.python_score            = 'python'

params.publish_intermediates   = false

input_path  = params.input ?: params.input_fasta
models_path = params.models_dir ?: params.model_dir

if( !input_path ) {
    error "Please provide --input or --input_fasta <proteins.faa[.gz]>"
}

if( !models_path ) {
    error "Please provide --models_dir or --model_dir <checkpoint directory>"
}

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
    set -euo pipefail

    ${params.python_cpu} ${projectDir}/bin/amrfold_split_fasta.py \\
      --input ${input_fasta} \\
      --outdir shards \\
      --manifest manifest.tsv \\
      --seqs-per-shard ${params.chunk_size}
    """
}

process EXTRACT_FEATURES_GROUP {
    tag { "group_${group_id}" }
    label 'extract'

    publishDir "${params.outdir}/features_lmdb", mode: 'copy', enabled: params.publish_intermediates

    input:
    tuple val(group_id), path(shard_fastas)

    output:
    tuple val(group_id), path('group_work'), emit: group_features

    script:
    def prost = params.prostt5_dir ? "--prostt5_dir ${params.prostt5_dir}" : ""
    def localOnly = params.local_files_only ? "--local_files_only true" : "--local_files_only false"

    """
    set -euo pipefail

    export PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

    GROUP_GPU=\$(( ${group_id} - 1 ))

    export CUDA_VISIBLE_DEVICES="\${GROUP_GPU}"
    export NVIDIA_VISIBLE_DEVICES="\${GROUP_GPU}"
    export SINGULARITYENV_CUDA_VISIBLE_DEVICES="\${GROUP_GPU}"
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="\${GROUP_GPU}"
    export SINGULARITYENV_NVIDIA_VISIBLE_DEVICES="\${GROUP_GPU}"
    export APPTAINERENV_NVIDIA_VISIBLE_DEVICES="\${GROUP_GPU}"

    export HF_HOME="\${TMPDIR:-/tmp}/hf_home"
    export TRANSFORMERS_CACHE="\${TMPDIR:-/tmp}/hf_cache"
    export HUGGINGFACE_HUB_CACHE="\${TMPDIR:-/tmp}/hf_hub"
    mkdir -p "\${HF_HOME}" "\${TRANSFORMERS_CACHE}" "\${HUGGINGFACE_HUB_CACHE}"

    echo "[EXTRACT_FEATURES_GROUP] Host: \$(hostname)"
    echo "[EXTRACT_FEATURES_GROUP] Group: ${group_id}"
    echo "[EXTRACT_FEATURES_GROUP] Assigned physical GPU: \${GROUP_GPU}"
    echo "[EXTRACT_FEATURES_GROUP] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}"
    echo "[EXTRACT_FEATURES_GROUP] TMPDIR=\${TMPDIR:-unset}"
    echo "[EXTRACT_FEATURES_GROUP] HF_HOME=\${HF_HOME}"
    nvidia-smi || true

    mkdir -p group_work/shards group_work/features

    cp *.fa group_work/shards/

    printf 'shard\\tn_sequences\\n' > group_work/extract_manifest.tsv

    for f in group_work/shards/*.fa; do
      n=\$(grep -c '^>' "\$f" || true)
      printf '%s\\t%s\\n' "\$(basename "\$f")" "\$n" >> group_work/extract_manifest.tsv
    done

    ${params.python_gpu} ${projectDir}/bin/extract_features_manifest.py \\
      --manifest group_work/extract_manifest.tsv \\
      --shards_dir group_work/shards \\
      --lmdb_root group_work/features \\
      --map_size_gb ${params.map_size_gb} \\
      --device ${params.device} \\
      --half_precision ${params.half_precision} \\
      --batch_size ${params.extract_batch_size} \\
      --token_budget ${params.extract_token_budget} \\
      --commit_every ${params.extract_commit_every} \\
      --skip_completed true \\
      ${localOnly} \\
      ${prost}
    """
}

process SCORE_LMDB_GROUP {
    tag { "group_${group_id}" }
    label 'score'

    publishDir "${params.outdir}/shards", mode: 'copy', enabled: true, pattern: '*.predictions.tsv'

    input:
    tuple val(group_id), path(group_work)
    path checkpoints

    output:
    path '*.predictions.tsv', emit: group_predictions

    script:
    def ckpts = checkpoints.collect { it.getName() }.join(' ')

    """
    set -euo pipefail

    export PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

    GROUP_GPU=\$(( ${group_id} - 1 ))

    export CUDA_VISIBLE_DEVICES="\${GROUP_GPU}"
    export NVIDIA_VISIBLE_DEVICES="\${GROUP_GPU}"
    export SINGULARITYENV_CUDA_VISIBLE_DEVICES="\${GROUP_GPU}"
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="\${GROUP_GPU}"
    export SINGULARITYENV_NVIDIA_VISIBLE_DEVICES="\${GROUP_GPU}"
    export APPTAINERENV_NVIDIA_VISIBLE_DEVICES="\${GROUP_GPU}"

    export HF_HOME="\${TMPDIR:-/tmp}/hf_home"
    export TRANSFORMERS_CACHE="\${TMPDIR:-/tmp}/hf_cache"
    export HUGGINGFACE_HUB_CACHE="\${TMPDIR:-/tmp}/hf_hub"
    mkdir -p "\${HF_HOME}" "\${TRANSFORMERS_CACHE}" "\${HUGGINGFACE_HUB_CACHE}"

    echo "[SCORE_LMDB_GROUP] Host: \$(hostname)"
    echo "[SCORE_LMDB_GROUP] Group: ${group_id}"
    echo "[SCORE_LMDB_GROUP] Assigned physical GPU: \${GROUP_GPU}"
    echo "[SCORE_LMDB_GROUP] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}"
    echo "[SCORE_LMDB_GROUP] TMPDIR=\${TMPDIR:-unset}"
    nvidia-smi || true

    ${params.python_score} ${projectDir}/bin/score_lmdb_manifest_ensemble.py \\
      --manifest ${group_work}/extract_manifest.tsv \\
      --shards_dir ${group_work}/shards \\
      --lmdb_root ${group_work}/features \\
      --output_dir . \\
      --device ${params.score_device} \\
      --batch_size ${params.infer_batch_size} \\
      --num_workers ${params.num_workers} \\
      --bin_threshold ${params.bin_threshold} \\
      --gate_classes ${params.gate_classes} \\
      --gate_tau ${params.gate_tau} \\
      --gate_delta ${params.gate_delta} \\
      --skip_completed true \\
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
    set -euo pipefail

    ${params.python_cpu} ${projectDir}/bin/amrfold_merge_predictions.py \\
      --output amrfold_predictions.tsv ${inputs}
    """
}

workflow {
    ch_input = Channel.fromPath(input_path, checkIfExists: true)

    ch_ckpts = Channel
        .fromPath("${models_path}/best_checkpoint_seed_*.pt", checkIfExists: true)
        .ifEmpty {
            error "No checkpoints found under ${models_path}"
        }
        .collect()

    split_res = SPLIT_FASTA(ch_input)

    ch_shard_groups = split_res.shards
        .flatten()
        .collect()
        .flatMap { shard_list ->
            def n_groups = Math.max(1, params.extract_groups as int)
            def groups = (0..<n_groups).collect { [] }

            shard_list
                .sort { a, b -> a.getName() <=> b.getName() }
                .eachWithIndex { f, i ->
                    groups[i % n_groups] << f
                }

            groups
                .withIndex()
                .findAll { item -> item[0].size() > 0 }
                .collect { item ->
                    tuple(item[1] + 1, item[0])
                }
        }

    extract_res = EXTRACT_FEATURES_GROUP(ch_shard_groups)

    if( params.score_after_all_extract ) {
        ch_groups_for_score = extract_res.group_features
            .collect(flat: false)
            .flatMap { collected_groups ->
                collected_groups.collect { item ->
                    tuple(item[0], item[1])
                }
            }
    } else {
        ch_groups_for_score = extract_res.group_features
    }

    score_res = SCORE_LMDB_GROUP(ch_groups_for_score, ch_ckpts)

    MERGE_PREDICTIONS(
        score_res.group_predictions
            .flatten()
            .collect()
    )
}
