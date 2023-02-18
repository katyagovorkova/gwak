from re import L
import numpy as np
import os
import matplotlib.pyplot as plt
from anomaly.evaluation.kde import KDE

def make_real_QUAK(datae, plot_path, class_labels):
    #NOTE USE OF NOISE, DATA HERE IS MEANINGLESS
    #EVERYTHING IS ORDERED THE SAME AS class_labels
    noise_data, glitch_data = datae
    noise_evals = noise_data
    glitch_evals = glitch_data

    #"fake" QUAK space - noise_evals plotted against glitch_evals
    plt.figure(figsize=(17, 10))

    plt.scatter(noise_evals[:, 0], noise_evals[:, 1], label = class_labels[0])
    plt.scatter(glitch_evals[:, 0], glitch_evals[:, 1], label = class_labels[1])

    plt.xlabel(f"{class_labels[0]} loss", fontsize = 15)
    plt.ylabel(f"{class_labels[1]} loss", fontsize=15)
    plt.loglog()
    plt.legend()
    plt.title("Distribution of points in REAL quak space", fontsize=17)
    plt.savefig(f"{plot_path}/real_quak.png", dpi=300)
    plt.show()

#doing predictions on validation data for now
def make_fake_QUAK(datae, KDES, plot_path, class_labels):
    noise_data, glitch_data = datae
    noise_kde, glitch_kde = KDES

    noise_evals = np.vstack([glitch_kde.predict(noise_data, convert_ln=False),
                            noise_kde.predict(noise_data, convert_ln=False)]).T
    glitch_evals = np.vstack([glitch_kde.predict(glitch_data, convert_ln=False),
                            noise_kde.predict(glitch_data, convert_ln=False)]).T
    #print("glitch eval shape", glitch_evals.shape)

    def px(data):
        return -(data)
    noise_evals = px(noise_evals)
    glitch_evals = px(glitch_evals)

    xmin = min(np.min(noise_evals[:, 0]), np.min(glitch_evals[:, 0]))
    xmax = max(np.max(noise_evals[:, 0]), np.max(glitch_evals[:, 0]))
    ymin = min(np.min(noise_evals[:, 1]), np.min(glitch_evals[:, 1]))
    ymax = max(np.max(noise_evals[:, 1]), np.max(glitch_evals[:, 1]))

    print("x:", xmin, xmax)
    print("y:", ymin, ymax)

    #"fake" QUAK space - noise_evals plotted against glitch_evals
    plt.figure(figsize=(17, 10))
    #horizontal 1/2 line
    half = -np.log(0.5)
    plt.plot([xmin, xmax], [half, half], c="blue", label = "half confidence noise")
    #vertical
    plt.plot([half, half], [ymin, ymax], c="orange", label = "half confidence glitch")
    plt.scatter(noise_evals[:, 0], noise_evals[:, 1], label = "noise")
    plt.scatter(glitch_evals[:, 0], glitch_evals[:, 1], label = "glitch")

    plt.xlabel(f"-ln(P({class_labels[0]}))", fontsize = 15)
    plt.ylabel(f"-ln(P({class_labels[1]}))", fontsize=15)
    plt.loglog()
    plt.legend()
    plt.title("Distribution of points in fake quak space", fontsize=17)
    plt.savefig(f"{plot_path}/fake_quak.png", dpi=300)
    plt.show()

def make_ROC(kind, datae, KDES, plot_path, class_labels):
    #go with class0 being noise for now
    noise_data, glitch_data = datae
    noise_kde, glitch_kde = KDES

    noise_evals = np.subtract(glitch_kde.predict(noise_data, convert_ln=False),
                            noise_kde.predict(noise_data, convert_ln=False))
    glitch_evals = np.subtract(glitch_kde.predict(glitch_data, convert_ln=False),
                            noise_kde.predict(glitch_data, convert_ln=False))

    a = min(np.min(noise_evals), np.min(glitch_evals))
    b = max(np.max(noise_evals), np.max(glitch_evals))
    #print(a, b)
    #ROC_bars = np.logspace(np.log10(a), np.log10(b), num=1000, base=10)
    ROC_bars = np.linspace(-100, 100, 10000)
    #making a ROC plot
    ROC_FPR = []
    ROC_TPR = []

    for bar in ROC_bars:
        FPR = len(noise_evals[noise_evals>=bar])/len(noise_evals)
        TPR = len(glitch_evals[glitch_evals>=bar])/len(glitch_evals)
        #print("roc calculation", bar, FPR, TPR)
        ROC_FPR.append(FPR)
        ROC_TPR.append(TPR)

    #print(ROC_FPR)
    #print(ROC_TPR)
    #calculate AUC
    #assuming strictly increasing
    AUC = 0
    ROC_TPR = [x for _,x in sorted(zip(ROC_FPR, ROC_TPR))]
    ROC_FPR = sorted(ROC_FPR)
    for i in range(len(ROC_FPR)-1):
        dx = ROC_FPR[i+1] - ROC_FPR[i]
        y = ROC_TPR[i]
        #print(dx, y)
        AUC += dx*y

    plt.figure(figsize = (17, 10))
    plt.title(f"ROC curve for glitches vs noise, AUC={AUC:.4f}")
    plt.plot(ROC_FPR, ROC_TPR)
    #plt.loglog()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(f"{plot_path}/{kind}_ROC.png", dpi=300)
    plt.show()


def calc_AUC(x, y):
    p = x.argsort()
    x = x[p]
    y = y[p]
    diff = np.diff(x)
    return np.dot(diff, y[:-1])
    
def multiclass_ROC(class_label, QUAK_datae_eval, KDE_models, plot_path):
    #run throght the trained KDE on all the data
    KDE_class = KDE_models[class_label]
    evals = dict()
    for key in QUAK_datae_eval:
        #print("KEY, CLASS_LABEL", key, class_label)
        if key != class_label: 
            evals[key] = KDE_class.predict(QUAK_datae_eval[key], convert_ln=False) #will give results as ln(P)
            #print("Key, evals", key, evals[key])
    main_evals = KDE_class.predict(QUAK_datae_eval[class_label], convert_ln=False)
    #now the plot

    #assert False
    #now calculate the FPR and TPR rates
    bars = np.linspace(-50, 50, num=5000)

    eval_keys = list(evals.keys())
    roc_vals = []
    for bar in bars:
        temp = []
        for key in eval_keys:
            dset = evals[key]
            #print("dset", dset.shape)
            temp.append(len(dset[dset>bar])/len(dset))
        roc_vals.append(temp)
    roc_vals = np.array(roc_vals)

    main_class_tpr = []
    for bar in bars:
        main_class_tpr.append(len(main_evals[main_evals>bar])/len(main_evals))
    main_class_tpr = np.array(main_class_tpr)
    print("SHAPES 152", main_class_tpr.shape, roc_vals.shape)

    plt.figure(figsize=(17/1.4, 10/1.4))

    for i, key in enumerate(eval_keys):
        #AUC = calc_AUC(roc_vals[:, i], main_class_tpr)
        #np.save(f"{plot_path}/C2C_ROC/ex_roc.npy", roc_vals[:, i])
        #np.save(f"{plot_path}/C2C_ROC/ex_tpr.npy", main_class_tpr)
        plt.plot(roc_vals[:, i], main_class_tpr, label=f"{key}")
        np.save(f"{plot_path}/C2C_ROCS/{class_label}_{key}.npy", np.stack([roc_vals[:, i], main_class_tpr]))
    plt.ylabel("TPR", fontsize=15)
    plt.xlabel("FPR", fontsize=15)
    #plt.yscale("log")
    plt.xscale("log")
    #plt.loglog()
    plt.title(f"ROC plot for {class_label}", fontsize=20)
    plt.legend()
    plt.savefig(f"{plot_path}/C2C_ROCS/{class_label}.png", dpi=300)
    plt.show()
    

def main(KDE_train_data_path, test_data_path, plot_path, class_labels, do_LS):
    LS_KDES = []
    KDE_models = dict()
    for data_class in class_labels:
        if do_LS: LS_data = np.load(f"{KDE_train_data_path}/{data_class}/LS_evals.npy")
        QUAK_data = np.load(f"{KDE_train_data_path}/{data_class}/QUAK_evals.npy")
       
        if do_LS: LS_KDES.append(KDE(LS_data, val_split=0.3, test_split=0))
        KDE_models[data_class] = (KDE(QUAK_data, val_split=0.15, test_split=0))
        
    LS_datae = []
    QUAK_datae = []
    QUAK_datae_eval = dict()
    for data_class in class_labels:
        #print("data_class", data_class)
        if do_LS: LS_data = np.load(f"{test_data_path}/{data_class}/LS_evals.npy")
        #QUAK_data = np.load(f"{test_data_path}/{data_class}/QUAK_evals.npy")
        QUAK_datae.append(np.load(f"{KDE_train_data_path}/{data_class}/QUAK_evals.npy"))

        
        #slx = int(len(QUAK_data)*0.7)
        #slx=0
        #if do_LS: LS_datae.append(LS_data[slx:])
        #QUAK_datae.append(QUAK_data[:slx])
        QUAK_datae_eval[data_class] = np.load(f"{test_data_path}/{data_class}/QUAK_evals.npy")

    #KDE_models = create_QUAK_models(class_labels, QUAK_datae)

    try:
        os.makedirs(f"{plot_path}/C2C_ROCS/")
    except FileExistsError:
        None

    for class_label in class_labels:
        multiclass_ROC(class_label, QUAK_datae_eval, KDE_models, plot_path)