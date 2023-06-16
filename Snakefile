modelclasses = ['bbh', 'sg', 'background', 'glitch', 'timeslides']
dataclasses = ['bbh_fm_optimization','sg_fm_optimization', 'bbh_varying_snr', 'sg_varying_snr']
wildcard_constraints:
    modelclass = '|'.join([x for x in modelclasses]),
    dataclass = '|'.join([x for x in dataclasses+modelclasses])

rule find_valid_segments:
    input:
        hanford_path = 'data/O3a_Hanford_segments.json',
        livingston_path = 'data/O3a_Livingston_segments.json'
    output:
        save_path = 'output/O3a_intersections.npy'
    script:
        'scripts/segments_intersection.py'

rule python_omicron:
    input:
        intersections = rules.find_valid_segments.output.save_path
    params:
        user_name = 'katya.govorkova',
        folder = directory('output/omicron/')
    shell:
        'mkdir -p {params.folder}; '
        'ligo-proxy-init {params.user_name}; '
        'python3 scripts/run_omicron.py {input.intersections} {params.folder}'

rule run_omicron:
    shell:
        'snakemake -c4 --keep-going python_omicron'

rule fetch_site_data:
    input:
        omicron = rules.run_omicron.output,
        intersections = rules.find_valid_segments.output.save_path
    output:
        'tmp/dummy_{site}.txt'
    shell:
        'touch {output}; '
        'python3 scripts/fetch_data.py {input.omicron} {input.intersections}\
            --site {wildcards.site}'

rule generate_dataset:
    input:
        omicron = 'output/omicron/',
        dependencies = expand(rules.fetch_site_data.output, site=['L1', 'H1'])
    output:
        file = 'output/data/{dataclass}.npy',
    shell:
        'python3 scripts/generate.py {input.omicron} {output.file} \
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
    # input:
        # data = expand(rules.pre_processing_step.output.train_file, dataclass='{dataclass}')
    params:
        data = [
            '/home/ryan.raikman/s23/gwak/data/train/background.npy',
            '/home/ryan.raikman/s23/gwak/data/train/bbh.npy',
            '/home/ryan.raikman/s23/gwak/data/train/glitch.npy',
            '/home/ryan.raikman/s23/gwak/data/train/sg.npy',
            ]
    output:
        savedir = directory('output/trained/{dataclass}'),
        model_file = 'output/trained/models/{dataclass}.pt'
    shell:
        'mkdir -p {output.savedir}; '
        'python3 scripts/train_quak.py {params.data} {output.model_file} {output.savedir}'

rule generate_timeslides_for_final_metric_train:
    input:
        # data_path = expand(rules.generate_dataset.output.file, dataclass='timeslides'),
        model_path = expand(rules.train_quak.output.model_file, dataclass=['bbh', 'sg', 'background', 'glitch'])
    params:
        data_path = '/home/ryan.raikman/s23/gwak/data/timeslides_segs.npy',
        shorten_timeslides = False
    output:
        save_folder_path = directory('output/timeslides/')
    shell:
        'mkdir -p {output.save_folder_path}; '
        'python3 scripts/evaluate_timeslides.py {params.data_path} {output.save_folder_path} {input.model_path} \
            --fm-shortened-timeslides {params.shorten_timeslides}'

rule evaluate_signals:
    input:
        # source_file = expand(rules.generate_dataset.output.file, dataclass='{signal_dataclass}'),
        model_path = expand(rules.train_quak.output.model_file, dataclass=['bbh', 'sg', 'background', 'glitch'])
    params:
        source_file = lambda w: '/home/ryan.raikman/s23/gwak/data/{w.signal_dataclass}_segs.npy'
    output:
        save_file = 'output/evaluated/{signal_dataclass}_evals.npy',
    shell:
        'python3 scripts/evaluate_data.py {params.source_file} {output.save_file} {input.model_path}'

rule train_final_metric:
    input:
        signals = expand(rules.evaluate_signals.output.save_file, signal_dataclass=['bbh_fm_optimization']),
        # timeslides = expand('output/timeslides/timeslide_evals_{i}.npy', i=[1, 2, 3])
    params:
        timeslides = expand('output/timeslides/timeslide_evals_{i}.npy', i=[1, 2, 3])
    output:
        params_file = 'output/trained/final_metric_params.npy'
    shell:
        'python3 scripts/final_metric_optimization.py {output.params_file} \
        --timeslide-path {params.timeslides} \
        --signal-path {input.signals}'

rule compute_far:
    input:
        # data_path = expand(rules.generate_dataset.output.file, dataclass='timeslides'),
        model_path = expand(rules.train_quak.output.model_file, dataclass=['bbh', 'sg', 'background', 'glitch']),
        metric_coefs_path = rules.train_final_metric.output.params_file
    params:
        data_path = '/home/ryan.raikman/s23/gwak/data/timeslides_segs.npy',
        shorten_timeslides = False
    output:
        save_path = 'output/far_bins.npy'
    shell:
        'python3 scripts/evaluate_timeslides.py {params.data_path} {output.save_path} {input.model_path} \
            --metric-coefs-path {input.metric_coefs_path} \
            --fm-shortened-timeslides {params.shorten_timeslides}'

rule quak_plotting_prediction_and_recreation:
    input:
        model_path = expand(rules.train_quak.output.model_file, dataclass=['bbh', 'sg', 'background', 'glitch']),
        # test_data = expand(rules.pre_processing_step.output.test_file, dataclass='{dataclass}')
    params:
        test_data = lambda w: '/home/ryan.raikman/s23/gwak/data/test/{dataclass}.npy'
        reduce_loss = False
    output:
        save_file = 'output/evaluated/quak_{dataclass}.npz'
    shell:
        'python3 scripts/quak_predict.py {params.test_data} {output.save_file} {params.reduce_loss} \
            --model-path {input.model_path} '

rule all_quak_pr:
    input:
        expand(rules.quak_plotting_prediction_and_recreation.output.save_file, dataclass=['bbh', 'sg', 'background', 'glitch'])

rule plot_results:
    input:
        dependencies = rules.compute_far.output.save_path
    params:
        evaluation_dir = 'output/',
    output:
        save_path = directory('output/plots/')
    shell:
        'mkdir -p {output.save_path}; '
        'python3 scripts/plotting.py {params.evaluation_dir} {output.save_path}'

rule make_pipeline_plot:
    shell:
        'snakemake plot_results --dag | dot -Tpdf > dag.pdf'