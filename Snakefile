rule run_omicron:
    input:
        script = 'scripts/run_omicron.py'
    output:
        folder_path = directory('output/omicron/')
    shell:
        'python3 {input.script} {output.folder_path}'

rule generate_dataset:
    input:
        script = 'scripts/generate.py',
        omicron = rules.run_omicron.output.folder_path
    output:
        file = 'output/data/{dataclass}_segs.npy',
    shell:
        'python3 {input.script} {output.file} {input.omicron}\
            --stype {wildcards.dataclass}'

rule pre_processing_step:
    input:
        script = 'scripts/pre_processing.py',
        file = lambda wildcards: expand(rules.generate_dataset.output.file, dataclass={wildcards.dataclass})
    output:
        train_file = 'output/data/train/{dataclass}.npy',
        test_file = 'output/data/test/{dataclass}.npy'
    shell:
        'python3 {input.script} {input.file} {output.train_file} {output.test_file}'

rule train_quak:
    input:
        script = 'scripts/train_quak.py',
    params:
        data = lambda wildcards: f'data/TRAIN_PROCESS/{wildcards.dataclass}.npy'
    output:
        savedir = directory('output/trained/{dataclass}')
    shell:
        'mkdir -p {output.savedir}; '
        'python3 {input.script} {params.data} {output.savedir}'

rule train_all_quak:
    input:
        expand(rules.train_quak.output.savedir, dataclass=['bbh', 'sg', 'bkg', 'glitch'])

rule data_prediction:
    input:
        script = 'scripts/predict.py'
        # datae = []
        # #class_labels = []
        # for file in sorted(os.listdir(f"{config['save_path']}/DATA/TEST_PROCESS/")):
        #     datae.append(np.load(f"{config['save_path']}/DATA/TEST_PROCESS/" + file))
        #     #class_labels.append(file[:-4])

    output:
    shell:
        'python3 {input.script}'
        # predict_main(datae,
        #             f"{config['save_path']}/TRAINED_MODELS/",
        #             f"{config['save_path']}/DATA_PREDICTION/TEST/",
        #             class_labels,
        #         V['train_LS'])

rule plotting:
    input:
        script = 'scripts/plotting.py'
    output:
    shell:
        'python3 {input.script}'
        # plotting_main(f"{config['save_path']}/DATA_PREDICTION/TEST/",
        #               f"{config['save_path']}/PLOTS/",
        #               class_labels,
        #               True,
        #               V['train_LS'])

rule ae_prediction:
    input:
        script = 'scripts/autoencoder_prediction.py'
        # autoencoder_prediction_main(config['save_path'], V['train_LS'])

rule nn_quak_runthrough:
    input:
        script = 'scripts/nn_quak_runthrough.py'
        # nn_quak_runthrough_main(config['save_path'])

rule data_runthrough:
    input:
        script = 'scripts/full_data_runthrough.py'
        # runthrough_main(V['runthrough_path'], config['save_path'], 5, kde_models, NN_quak=True)

# rule calculate_pearson:
#     input:
#     output:
#     shell:

# rule calculate_metric:
#     input:
#     output:
#     shell:

# rule plot_results:
#     input:
#     output:
#     shell: