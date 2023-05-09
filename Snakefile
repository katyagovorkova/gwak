configfile: 'config.yaml'


rule generate_background:
    input:
        script = 'generate.py',
    params:
        stype = 'background'
    output:
        file = 'output/background_segs.npy'
    shell:
        'python {input.script} {output.file} {params.stype}'

rule generate_bbh:
    input:
        script = 'generate.py',
    params:
        stype = 'bbh'
    output:
        file = 'output/bbh_segs.npy'
    shell:
        'python {input.script} {output.file} {params.stype}'

rule generate_sg:
    input:
        script = 'generate.py',
    params:
        stype = 'sg'
    output:
        file = 'output/sg_segs.npy'
    shell:
        'python {input.script} {output.file} {params.stype}'

rule generate_glitch:
    input:
        script = 'generate.py',
    params:
        stype = 'glitch'
    output:
        file = 'output/glitch_segs.npy'
    shell:
        'python {input.script} {output.file} {params.stype}'

rule generate_data:
    output:
        rules.generate_background.output,
        rules.generate_bbh.output,
        rules.generate_sg.output,
        rules.generate_glitch.output

# rule train_quak:
#     input:
#     output:
#     shell:

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

rule train_test_split:
    input:
        script = 'train_test_split.py'
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
        script = 'libs/pre_processing/anomaly/pre_processing/pre_processing.py',
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

rule train_LS_main:
    input:
        script = 'anomaly/evaluation/train_LS_main.py'
    output:
    shell:
        'python3 {script}'

        # train_LS_main(datae,
        #             V['network_type'],
        #             f"{trained_model_path}/LS/",
        #             V['batch_size'],
        #             V['epochs'],
        #             V['bottleneck'],
        #             f"{config['save_path']}/DATA/TEST_PROCESS/")
        # #don't want it to run through the rest of the analysis, since something will break
        # assert False

rule train_QUAK_main:
    input:
        script = 'anomaly/evaluation/train_QUAK_main.py'

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

rule eval_data_prediction_step:
    input:
        script = 'anomaly/evaluation/predict_main.py'
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

rule eval_plotting_step:
    input:
        script = 'anomaly/evaluation/plotting_main.py'
    output:
    shell:
        'python3 {script}'
        # plotting_main(f"{config['save_path']}/DATA_PREDICTION/TEST/",
        #               f"{config['save_path']}/PLOTS/",
        #               class_labels,
        #               True,
        #               V['train_LS'])

rule roc_plotting_step:
    input:
        script = 'anomaly/evaluation/kde_main.py'
    output:
    shell:
        'python3 {script}'
        # kde_main(f"{config['save_path']}/DATA_PREDICTION/TRAIN/",
        #         f"{config['save_path']}/DATA_PREDICTION/TEST/",
        #         f"{config['save_path']}/PLOTS/",
        #         class_labels,
        #         V['train_LS'])

rule ae_prediction_step:
    input:
        script = 'anomaly/evaluation/autoencoder_prediction_main.py'
        # autoencoder_prediction_main(config['save_path'], V['train_LS'])

rule nn_quak_runthrough_step:
    input:
        script = 'anomaly/evaluation/nn_quak_runthrough_main.py'
        # nn_quak_runthrough_main(config['save_path'])

rule data_runthrough_step:
    input:
        script = 'anomaly/evaluation/runthrough_main.py'
        # runthrough_main(V['runthrough_path'], config['save_path'], 5, kde_models, NN_quak=True)
