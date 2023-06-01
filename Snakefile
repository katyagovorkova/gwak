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
    input:
        # omicron = rules.run_omicron.output
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
    # params:
    #     data = expand(rules.pre_processing_step.output.train_file, dataclass='{dataclass}')
    output:
        savedir = directory('output/trained/{dataclass}')
    shell:
        'mkdir -p {output.savedir}; '
        'python3 scripts/train_quak.py {input.data} {output.savedir}'

rule train_all_quak:
    input:
        expand(rules.train_quak.output.savedir, dataclass=['bbh', 'sg', 'background', 'glitch'])

rule ae_prediction:
    input:
        model_path = expand(f'{rules.train_quak.output.savedir}/ae.h5', dataclass='{modelclass}'),
        test_data = expand(rules.pre_processing_step.output.test_file, dataclass='{dataclass}')
    # params:
    #     test_data = lambda wildcards: f'output/data/test/{wildcards.dataclass}.npy'
    output:
        save_file = 'output/evaluated/model_{modelclass}/{dataclass}.npy'
    shell:
        'mkdir -p output/evaluated/model_{wildcards.modelclass}/; '
        'python3 scripts/predict.py {input.test_data} {input.model_path} {output.save_file}'

rule merge_ae_predictions:
    input:
        bbh = 'output/evaluated/model_bbh/{dataclass}.npy',
        sg = 'output/evaluated/model_sg/{dataclass}.npy',
        background = 'output/evaluated/model_background/{dataclass}.npy',
        glitch = 'output/evaluated/model_glitch/{dataclass}.npy',
    output:
        'output/evaluated/quak_{dataclass}.npz'
    shell:
        'python3 scripts/merge_ae_predictions.py {input.bbh} {input.sg} {input.glitch} {input.background} {output}'

rule calculate_pearson:
    input:
        # data_path = 'output/data/test/{dataclass}.npy'
    params:
        data_path = lambda wildcards: "/home/ryan.raikman/s22/anomaly/generated_timeslides/1241093492_1241123810/timeslide_data.npy" \
            if 'timeslides' in wildcards.dataclass else 'output/data/test/{dataclass}.npy'
    output:
        save_file = 'output/evaluated/pearson_{dataclass}.npy'
    shell:
        'mkdir -p output/data/test/correlations/; '
        'python3 scripts/pearson.py {params.data_path} {output.save_file}'

rule train_metric:
    input:
        bbh_quak = 'output/evaluated/quak_bbh.npz',
        bbh_pearson = 'output/evaluated/pearson_bbh.npy',
        sg_pearson = 'output/evaluated/pearson_sg.npy',
        sg_quak = 'output/evaluated/quak_sg.npz',
        timeslides_pearson = 'output/evaluated/pearson_timeslides.npy',
        timeslides_quak = 'output/evaluated/quak_timeslides.npz',
    output:
        params_file = 'output/trained/es_params.npy'
    shell:
        'python3 scripts/evolutionary_search.py'

rule plot_results:
    input:
        evaluation_dir = 'output/evaluated/'
    output:
        directory('output/plots/')
    shell:
        'mkdir -p {output}; '
        'python3 scripts/plotting.py {input.evaluation_dir} {output}'
