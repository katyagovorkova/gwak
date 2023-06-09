dataclasses = ['bbh', 'sg', 'background', 'glitch', 'timeslides']
wildcard_constraints:
    dataclass = '|'.join([x for x in dataclasses]),
    modelclass = '|'.join([x for x in dataclasses])

rule run_omicron:
    params:
        'output/'
    shell:
        'python3 scripts/run_omicron.py {params}'

rule fetch_site_data:
    params:
        lambda wildcards: directory(f'/home/ryan.raikman/s22/anomaly/data2/glitches/{wildcards.site}/data/')
    output:
        temp('tmp/dummy_{site}.txt')
    shell:
        'touch {output}; '
        'mkdir -p {params}; '
        'python3 scripts/fetch_data.py {params} {wildcards.site}'

rule fetch_data:
    input:
        expand(rules.fetch_site_data.output, site=['L1', 'H1'])

rule generate_dataset:
    # input:
    #     omicron = rules.run_omicron.output
    params:
        omicron = '/home/ryan.raikman/s22/anomaly/data2/glitches/'
    output:
        file = 'output/data/{dataclass}_segs.npy',
    shell:
        'python3 scripts/generate.py {params.omicron} {output.file} \
            --stype {wildcards.dataclass}'

rule pre_processing_step:
    input:
        file = expand(rules.generate_dataset.output.file, dataclass='{dataclass}')
    output:
        train_file = 'output/data/train/{dataclass}.npy',
        test_file = 'output/data/test/{dataclass}.npy'
    shell:
        'python3 scripts/pre_processing.py {input.file} {output.train_file} {output.test_file}'

rule train_quak:
    input:
        data = expand(rules.pre_processing_step.output.train_file, dataclass='{dataclass}')
    output:
        savedir = directory('output/trained/{dataclass}'),
        model_file = 'output/trained/models/{dataclass}.pt'
    shell:
        'mkdir -p {output.savedir}; '
        'python3 scripts/train_quak.py {input.data} {output.model_file} {output.savedir}'

rule train_all_quak:
    input:
        expand(rules.train_quak.output.savedir, dataclass=['bbh', 'sg', 'background', 'glitch'])

rule quak_prediction:
    input:
        model_path = expand(rules.train_quak.output.model_file, dataclass=['bbh', 'sg', 'background', 'glitch']),
        test_data = expand(rules.pre_processing_step.output.test_file, dataclass='{dataclass}')
    output:
        save_file = 'output/evaluated/quak_{dataclass}.npz'
    shell:
        'python3 scripts/quak_predict.py {input.test_data} {output.save_file} True \
            --model-path {input.model_path} '

rule quak_recreation:
    params:
        reduce_loss = False
    input:
        model_path = expand(rules.train_quak.output.model_file, dataclass=['bbh', 'sg', 'background', 'glitch']),
        test_data = expand(rules.pre_processing_step.output.test_file, dataclass='{dataclass}')
    output:
        save_file = 'output/recreation/quak_{dataclass}.npz'
    shell:
        'python3 scripts/quak_predict.py {input.test_data} {output.save_file} {params.reduce_loss} \
         --model-path {input.model_path} '

rule all_quak_recreation:
    input:
        expand(rules.quak_recreation.output.save_file, dataclass=['bbh', 'sg', 'background', 'glitch'])

rule generate_fm_signals:
    params:
        omicron = '/home/ryan.raikman/s22/anomaly/data2/glitches/'
    output:
        save_file = 'output/fm_files/{fm_signal_dataclass}_injections.npy',
    shell:
        'python3 scripts/generate.py {params.omicron} {output.save_file} \
            --stype {wildcards.fm_signal_dataclass};'

rule evaluate_fm_signals:
    input:
        source_file = 'output/fm_files/{fm_signal_dataclass}_fm_optimization_injections.npy',
        model_path = expand(rules.train_quak.output.model_file, dataclass=['bbh', 'sg', 'background', 'glitch'])
    output:
        save_file = 'output/fm_files_eval/signals/{fm_signal_dataclass}_evals.npy'
    shell:
        'python3 scripts/evaluate_data.py {input.source_file} {output.save_file} {input.model_path}'

rule generate_timeslides_for_final_metric_train:
    input:
        data_path = 'output/data/timeslides_segs.npy',
        model_path = expand(rules.train_quak.output.model_file, dataclass=['bbh', 'sg', 'background', 'glitch'])
    output:
        save_folder_path = directory('output/fm_files_eval/timeslides/')
    shell:
        'mkdir output/fm_files_eval/timeslides/ ;'
        'python3 scripts/evaluate_timeslides.py {input.data_path} {output.save_folder_path} {input.model_path} \
            --fm_shortened_timeslide True'

rule train_metric:
    input:
        signals = expand(rules.evaluate_fm_signals.output.save_file, fm_signal_dataclass=['bbh', 'sg']),
        timeslides = expand('output/fm_files_eval/timeslides/timeslide_evals_{i}.npy', i=[1])
    output:
        params_file = 'output/trained/final_metric_params.npy'
    shell:
        'python3 scripts/final_metric_optimization.py {input.timeslides} {output.params_file} {input.signals}'


rule calculate_pearson:
    input:
        data_path = 'output/data/test/{dataclass}.npy'
    output:
        save_file = 'output/evaluated/pearson_{dataclass}.npy'
    shell:
        'mkdir -p output/data/test/correlations/; '
        'python3 scripts/pearson.py {input.data_path} {output.save_file}'

rule plot_results:
    input:
        dependencies = rules.train_metric.output.params_file
    params:
        evaluation_dir = 'output/',
    output:
        directory('output/plots/')
    shell:
        'mkdir -p {output}; '
        'python3 scripts/plotting.py {params.evaluation_dir} {output}'

rule make_pipeline_plot:
    shell:
        'snakemake plot_results --dag | dot -Tpdf > dag.pdf'