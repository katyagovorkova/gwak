dataclass = ['bbh', 'sg', 'background', 'glitch']
wildcard_constraints:
    dataclass = '|'.join([x for x in dataclass]),
    modelclass = '|'.join([x for x in dataclass])

rule run_omicron:
    input:
        script = 'scripts/run_omicron.py'
    params:
        'output/'
    shell:
        'python3 {input.script} {params}'

rule fetch_site_data:
    input:
        script = 'scripts/fetch_data.py'
    params:
        lambda wildcards: directory(f'output/{wildcards.site}/data/')
    output:
        temp('tmp/dummy_{site}.txt')
    shell:
        'touch {output}; '
        'mkdir -p {params}; '
        'python3 {input.script} {params} {wildcards.site}'

rule fetch_data:
    input:
        expand(rules.fetch_site_data.output, site=['L1', 'H1'])

rule generate_dataset:
    input:
        script = 'scripts/generate.py',
        omicron = rules.run_omicron.output
    output:
        file = 'output/data/{dataclass}_segs.npy',
    shell:
        'python3 {input.script} {input.omicron} {output.file} \
            --stype {wildcards.dataclass}'

rule pre_processing_step:
    input:
        script = 'scripts/pre_processing.py',
        file = expand(rules.generate_dataset.output.file, dataclass='{dataclass}')
    output:
        train_file = 'output/data/train/{dataclass}.npy',
        test_file = 'output/data/test/{dataclass}.npy'
    shell:
        'python3 {input.script} {input.file} {output.train_file} {output.test_file}'

rule train_quak:
    input:
        script = 'scripts/train_quak.py',
        data = expand(rules.pre_processing_step.output.train_file, dataclass='{dataclass}')
    # params:
    #     data = expand(rules.pre_processing_step.output.train_file, dataclass='{dataclass}')
    output:
        savedir = directory('output/trained/{dataclass}')
    shell:
        'mkdir -p {output.savedir}; '
        'python3 {input.script} {input.data} {output.savedir}'

rule train_all_quak:
    input:
        expand(rules.train_quak.output.savedir, dataclass=['bbh', 'sg', 'background', 'glitch'])

rule ae_prediction:
    input:
        script = 'scripts/predict.py',
        model_path = expand(f'{rules.train_quak.output.savedir}/ae.h5', dataclass='{modelclass}'),
        # test_data = expand(rules.pre_processing_step.output.test_file, dataclass='{dataclass}')
    params:
        test_data = lambda wildcards: f'output/data/test/{wildcards.dataclass}.npy'
    output:
        save_file = 'output/evaluated/model_{modelclass}/{dataclass}.npy'
    shell:
        'mkdir -p output/evaluated/model_{wildcards.modelclass}/; '
        'python3 {input.script} {params.test_data} {input.model_path} {output.save_file}'

rule all:
    input:
        expand(rules.ae_prediction.output.save_file, modelclass=['bbh', 'sg', 'background', 'glitch'], dataclass=['bbh', 'sg', 'background', 'glitch'])

rule calculate_pearson:
    input:
        script = 'scripts/pearson.py',
        data_path = 'output/data/test/{dataclass}.npy'
    # params:
    #     data_path = 'output/data/test/{dataclass}.npy'
    output:
        save_file = 'output/data/test/correlations/{dataclass}.npy'
    shell:
        'mkdir -p output/data/test/correlations/; '
        'python3 {input.script} {input.data_path} {output.save_file}'

rule train_metric:
    input:
        script = 'scripts/evolutionary_search.py'
    output:
        params_file = 'output/trained/es_params.npy'
    shell:
        'python3 {input.script}'

rule plot_results:
    input:
        script = 'scripts/plotting.py',
        evaluation_dir = 'output/evaluated/'
    output:
        directory('output/plots/')
    shell:
        'mkdir -p {output}; '
        'python3 {input.script} {input.evaluation_dir} {output}'
