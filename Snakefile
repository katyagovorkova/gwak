configfile: 'config.yaml'


rule run_omicron:
    input:
        script = 'scripts/omicron.py'
    output:
        folder_path = directory('output/omicron/')
    shell:

rule generate_dataset:
    input:
        script = 'scripts/generate.py',
        omicron_output = rules.run_omicron.output.folder_path
    output:
        file = 'output/{dataset}_segs.npy',
    shell:
        'python3 {input.script} {input.omicron_output} {output.file} \
            --stype {wildcards.dataset}'

rule generate_all_data:
    input:
        expand(rules.generate_dataset.output.file, dataset=['bbh', 'sg', 'background', 'glitch'])

rule train_test_split:
    input:
        script = 'scripts/train_test_split.py'
    params:
        test_split = config['test_split'],
        data_path = config['data_path']
    output:
        train_dir = directory(config['train_path']),
        test_dir = directory(config['test_path'])
    shell:
        'mkdir -p {output.train_dir};'
        'mkdir -p {output.test_dir};'
        'python3 {input.script} {output.train_dir} {output.test_dir} \
            --test-split {params.test_split} \
            --data-path {params.data_path} '

rule pre_processing_step:
    input:
        script = 'scripts/pre_processing.py',
        train_dir = rules.train_test_split.output.train_dir,
        test_dir = rules.train_test_split.output.test_dir
    params:
        method = {config["data_preprocessing_method"]}
    output:
        train_dir_process = directory(config['train_process_path']),
        test_dir_process = directory(config['test_process_path'])
    shell:
        'python3 {input.script} {input.train_dir} \
                          {params.method} \
                          {output.train_dir_process}; '
        'python3 {input.script} {input.test_dir} \
                          {params.method} \
                          {output.test_dir_process}; '

rule train_quak:
    input:
        script = 'scripts/train_quak.py'

        # #load training data
        # datae = []

        # for file in sorted(os.listdir(f"{config['save_path']}/DATA/TRAIN_PROCESS/")):
        #     data = np.load(f"{config['save_path']}/DATA/TRAIN_PROCESS/" + file)
        #     print(f"loaded data from file: {file}, shape is: {data.shape}")
        #     datae.append(data)
        #     print("after process, indiv shape", data.shape)
        # trained_model_path = f"{config['save_path']}/TRAINED_MODELS/"
    output:
    shell:
        'python3 {script}'
            # train_QUAK_main(datae,
            #             V['network_type'],
            #             f"{trained_model_path}/QUAK/",
            #             V['batch_size'],
            #             V['epochs'],
            #             V['bottleneck'],
            #             class_labels)

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
        'python3 {script}'
        # predict_main(datae,
        #             f"{config['save_path']}/TRAINED_MODELS/",
        #             f"{config['save_path']}/DATA_PREDICTION/TEST/",
        #             class_labels,
        #         V['train_LS'])

rule eval_plotting:
    input:
        script = 'scripts/plotting.py'
    output:
    shell:
        'python3 {script}'
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