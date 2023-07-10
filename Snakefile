from config import VERSION

signalclasses = ['bbh', 'sg']
backgroundclasses = ['background', 'glitches']
modelclasses = signalclasses + backgroundclasses
dataclasses = [
    'timeslides',
    'bbh_fm_optimization',
    'sg_fm_optimization',
    'bbh_varying_snr',
    'sg_varying_snr',
    'wnb_varying_snr',
    'supernova_varying_snr']

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

rule generate_data:
    input:
        omicron = 'output/omicron/',
        intersections = rules.find_valid_segments.output.save_path,
    params:
        dependencies = expand(rules.fetch_site_data.output,
            site=['L1', 'H1'])
    output:
        file = 'output/data/{dataclass}.npz'
    shell:
        'python3 scripts/generate.py {input.omicron} {output.file} \
            --stype {wildcards.dataclass} \
            --intersections {input.intersections}'

rule upload_data:
    input:
        expand(rules.generate_data.output.file,
            dataclass='{dataclass}'),
    params:
        '/home/katya.govorkova/gwak/{version}/data/{dataclass}.npz'
    shell:
        'mkdir -p /home/katya.govorkova/gwak/{wildcards.version}/data/; '
        'cp {input} {params}; '

rule validate_data:
    input:
        expand(rules.upload_data.params,
            dataclass=modelclasses+dataclasses,
            version=VERSION)
    shell:
        'mkdir -p data/{VERSION}/; '
        'python3 scripts/validate_data.py {input}'

rule train_quak:
    input:
        data = expand(rules.upload_data.params,
            dataclass='{dataclass}',
            version=VERSION)
    output:
        savedir = directory('output/    trained/{dataclass}'),
        model_file = 'output/trained/models/{dataclass}.pt'
    shell:
        'mkdir -p {output.savedir}; '
        'python3 scripts/train_quak.py {input.data} {output.model_file} {output.savedir} '

rule recreation_and_quak_plots:
    input:
        models = expand(rules.train_quak.output.model_file,
            dataclass=modelclasses,
            version=VERSION),
        test_path = expand(rules.upload_data.params,
            dataclass='bbh',
            version=VERSION)
    output:
        savedir = directory('output/plots/')
    shell:
        'mkdir -p {output.savedir}; '
        'python3 scripts/rec_and_quak_plots.py {input.test_path} {input.models} {output.savedir}'

rule generate_timeslides_for_final_metric_train:
    input:
        data_path = expand(rules.upload_data.params,
            dataclass='timeslides',
            version=VERSION),
        model_path = expand(rules.train_quak.output.model_file,
            dataclass=modelclasses)
    params:
        shorten_timeslides = True
    output:
        save_folder_path = directory('output/timeslides/')
    shell:
        'mkdir -p {output.save_folder_path}; '
        'python3 scripts/evaluate_timeslides.py {input.data_path} {output.save_folder_path} {input.model_path} \
            --fm-shortened-timeslides {params.shorten_timeslides}'

rule evaluate_signals:
    input:
        source_file = expand(rules.upload_data.params,
            dataclass='{signal_dataclass}',
            version=VERSION),
        model_path = expand(rules.train_quak.output.model_file,
            dataclass=modelclasses)
    output:
        save_file = 'output/evaluated/{signal_dataclass}_evals.npy',
    shell:
        'python3 scripts/evaluate_data.py {input.source_file} {output.save_file} {input.model_path}'

rule train_final_metric:
    input:
        signals = expand(rules.evaluate_signals.output.save_file,
            signal_dataclass=['bbh_varying_snr', 'sg_varying_snr']),
        dependencies = rules.generate_timeslides_for_final_metric_train.output
    params:
        timeslides = expand('output/timeslides/timeslide_evals_{i}.npy',
            i=range(1, 140)),
        normfactors = expand('output/timeslides/normalization_params_{i}.npy',
            i=range(1, 140))
    output:
        params_file = 'output/trained/final_metric_params.npy',
        norm_factor_file = 'output/trained/norm_factor_params.npy'
    shell:
        'python3 scripts/final_metric_optimization.py {output.params_file} {output.norm_factor_file} \
            --timeslide-path {params.timeslides} \
            --signal-path {input.signals} \
            --norm-factor-path {params.normfactors}'

rule compute_far:
    input:
        data_path = expand(rules.upload_data.params,
            dataclass='timeslides',
            version=VERSION),
        model_path = expand(rules.train_quak.output.model_file,
            dataclass=modelclasses),
        metric_coefs_path = rules.train_final_metric.output.params_file,
        norm_factors_path = rules.train_final_metric.output.norm_factor_file
    params:
        shorten_timeslides = False
    output:
        save_path = 'output/far_bins.npy'
    shell:
        'python3 scripts/evaluate_timeslides.py {input.data_path} {output.save_path} {input.model_path} \
            --metric-coefs-path {input.metric_coefs_path} \
            --norm-factor-path {input.norm_factors_path} \
            --fm-shortened-timeslides {params.shorten_timeslides}'

rule quak_plotting_prediction_and_recreation:
    input:
        model_path = expand(rules.train_quak.output.model_file,
            dataclass=modelclasses),
        test_data = expand(rules.upload_data.params,
            dataclass='{dataclass}',
            version=VERSION)
    params:
        reduce_loss = False
    output:
        save_file = 'output/evaluated/quak_{dataclass}.npz'
    shell:
        'python3 scripts/quak_predict.py {input.test_data} {output.save_file} {params.reduce_loss} \
            --model-path {input.model_path} '

rule plot_results:
    input:
        rules.compute_far.output.save_path,
        'output/evaluated/bbh_varying_snr_evals.npy',
        'output/evaluated/sg_varying_snr_evals.npy',
        'output/evaluated/wnb_varying_snr_evals.npy',
        'output/evaluated/supernova_varying_snr_evals.npy'
    params:
        evaluation_dir = 'output/',
    output:
        save_path = directory('output/paper/')
    shell:
        'mkdir -p {output.save_path}; '
        'python3 scripts/plotting.py {params.evaluation_dir} {output.save_path}'

rule make_pipeline_plot:
    shell:
        'snakemake plot_results --dag | dot -Tpdf > dag.pdf'