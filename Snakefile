from config import VERSION
from config import SAMPLE_RATE, SEG_NUM_TIMESTEPS

models = ['lstm', 'dense', 'transformer']
modelclasses = ['bbh', 'sg', 'background', 'glitch']
dataclasses = ['timeslides', 'bbh_fm_optimization','sg_fm_optimization', 'bbh_varying_snr', 'sg_varying_snr', 'wnb_varying_snr']
wildcard_constraints:
    model = '|'.join([x for x in models]),
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

rule run_omicron:
    input:
        intersections = rules.find_valid_segments.output.save_path
    params:
        user_name = 'katya.govorkova',
        folder = 'output/omicron/'
    shell:
        'mkdir -p {params.folder}; '
        'ligo-proxy-init {params.user_name}; '
        'python3 scripts/run_omicron.py {input.intersections} {params.folder}'

rule fetch_site_data:
    input:
        omicron = rules.run_omicron.params.folder,
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
    output:
        file = 'output/data/{dataclass}.npy',
    shell:
        'python3 scripts/generate.py {input.omicron} {output.file} \
            --stype {wildcards.dataclass}'

rule pre_processing_step:
    input:
        file = expand(rules.generate_dataset.output.file,
            dataclass='{dataclass}')
    output:
        train_file = 'output/data/train/{dataclass}.npy',
        test_file = 'output/data/test/{dataclass}.npy'
    shell:
        'python3 scripts/pre_processing.py {input.file} {output.train_file} {output.test_file}'

rule upload_train_test_data:
    input:
        train_data = expand(rules.pre_processing_step.output.train_file,
            dataclass='{dataclass}'),
        test_data = expand(rules.pre_processing_step.output.test_file,
            dataclass='{dataclass}')
    params:
        train_data = '/home/ryan.raikman/s23/gwak/4096_200_curric/train/{dataclass}.npy',
        test_data = '/home/ryan.raikman/s23/gwak/4096_200_curric/test/{dataclass}.npy'
    shell:
        'mkdir -p /home/ryan.raikman/gwak/{wildcards.version}/train/; '
        'mkdir -p /home/ryan.raikman/gwak/{wildcards.version}/test/; '
        'cp {input.train_data} {params.train_data}; '
        'cp {input.test_data} {params.test_data}'

rule upload_generated_data:
    input:
        data = expand(rules.generate_dataset.output.file,
            dataclass='{dataclass}')
    params:
        data = '/home/ryan.raikman/gwak/{version}/data/{dataclass}.npy'
    shell:
        'mkdir -p /home/ryan.raikman/gwak/{wildcards.version}/data/; '
        'cp {input.data} {params.data}'

rule upload_data:
    input:
        expand(rules.upload_train_test_data.params,
            dataclass=modelclasses,
            version=VERSION),
        expand(rules.upload_generated_data.params,
            dataclass=dataclasses,
            version=VERSION)

rule train_quak:
    input:
        data = expand(rules.upload_train_test_data.params.train_data,
            dataclass='{dataclass}',
            version=VERSION)
    output:
        savedir = directory('output/{model}/trained/{dataclass}'),
        model_file = 'output/{model}/trained/models/{dataclass}.pt'
    shell:
        'mkdir -p {output.savedir}; '
        'python3 scripts/train_quak.py {input.data} {output.model_file} {output.savedir} \
            --model {wildcards.model}'

rule recreation_and_quak_plots:
    input: 
        model_path = 'output/{model}/trained/models/',
        test_path = expand(rules.upload_train_test_data.params.test_data, dataclass="bbh")
    output:
        savedir = directory('output/{model}/rec_and_quak/')
    shell:
        'mkdir -p {output.savedir};'
        'python3 scripts/rec_and_quak_plots.py {input.test_path} {input.model_path} {output.savedir}'


rule generate_timeslides_for_final_metric_train:
    input:
        data_path = expand(rules.generate_dataset.output.file,
            dataclass='timeslides',
            version=VERSION),
        model_path = expand(rules.train_quak.output.model_file,
            dataclass=modelclasses,
            model='{model}')
    params:
        shorten_timeslides = True
    output:
        save_folder_path = directory('output/{model}/timeslides/')
    shell:
        'mkdir -p {output.save_folder_path}; '
        'python3 scripts/evaluate_timeslides.py {input.data_path} {output.save_folder_path} {input.model_path} \
            --fm-shortened-timeslides {params.shorten_timeslides}'

rule evaluate_signals:
    input:
        source_file = expand(rules.generate_dataset.output.file,
            dataclass='{signal_dataclass}'),
        model_path = expand(rules.train_quak.output.model_file,
            dataclass=modelclasses,
            model='{model}')
    output:
        save_file = 'output/{model}/evaluated/{signal_dataclass}_evals.npy',
    shell:
        'python3 scripts/evaluate_data.py {input.source_file} {output.save_file} {input.model_path}'

rule train_final_metric:
    input:  
        signals = expand(rules.evaluate_signals.output.save_file,
            signal_dataclass=['bbh_fm_optimization', 'sg_fm_optimization'],
            model='{model}'),
    params:
        timeslides = expand('output/{model}/timeslides/timeslide_evals_{i}.npy',
            i=range(1, 70),
            model='{model}'),
        normfactors = expand('output/{model}/timeslides/normalization_params_{i}.npy',
            i=range(1, 70),
            model='{model}')
    output:
        params_file = 'output/{model}/trained/final_metric_params.npy',
        norm_factor_file = 'output/{model}/trained/norm_factor_params.npy'
    shell:
        'python3 scripts/final_metric_optimization.py {output.params_file} {output.norm_factor_file} \
        --timeslide-path {params.timeslides} \
        --signal-path {input.signals} \
        --norm-factor-path {params.normfactors}'

rule compute_far:
    input:
        data_path = expand(rules.generate_dataset.output.file,
            dataclass='timeslides',
            version=VERSION),
        model_path = expand(rules.train_quak.output.model_file,
            dataclass=modelclasses,
            model='{model}'),
        metric_coefs_path = expand(rules.train_final_metric.output.params_file,
            model='{model}'),
        norm_factors_path = expand(rules.train_final_metric.output.norm_factor_file,
            model='{model}')
    params:
        shorten_timeslides = False
    output:
        save_path = 'output/{model}/far_bins.npy'
    shell:
        'python3 scripts/evaluate_timeslides.py {input.data_path} {output.save_path} {input.model_path} \
            --metric-coefs-path {input.metric_coefs_path} --norm-factor-path {input.norm_factors_path} --fm-shortened-timeslides {params.shorten_timeslides}'

rule quak_plotting_prediction_and_recreation:
    input:
        model_path = expand(rules.train_quak.output.model_file,
            dataclass=modelclasses,
            model='{model}'),
        test_data = expand(rules.upload_train_test_data.params.test_data,
            dataclass='{dataclass}',
            version=VERSION)
    params:
        reduce_loss = False
    output:
        save_file = 'output/{model}/evaluated/quak_{dataclass}.npz'
    shell:
        'python3 scripts/quak_predict.py {input.test_data} {output.save_file} {params.reduce_loss} \
            --model-path {input.model_path} '

rule plot_model_results:
    input:
        dependencies = rules.compute_far.output.save_path
    params:
        evaluation_dir = 'output/{model}/',
    output:
        save_path = directory('output/{model}/plots/')
    shell:
        'mkdir -p {output.save_path}; '
        'python3 scripts/plotting.py {params.evaluation_dir} {output.save_path}'

rule plot_results:
    input:
        expand(rules.plot_model_results.output.save_path,
            model=models)

rule make_pipeline_plot:
    shell:
        'snakemake plot_results --dag | dot -Tpdf > dag.pdf'
