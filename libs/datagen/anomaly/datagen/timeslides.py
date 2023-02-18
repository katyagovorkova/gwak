import h5py
import os
import numpy as np
from gwpy.timeseries import TimeSeries

def find_h5(path):
    h5_file = None
    if not  os.path.exists(path): return None
    for file in os.listdir(path):
        if file[-3:] == ".h5":
            assert h5_file is None #make sure only 1 h5 file 
            h5_file = path + "/" + file

    assert h5_file is not None #did not find h5 file
    return h5_file

def clipping(seg:TimeSeries, sample_rate:int, clip_edge:int=1):
    clip_edge_datapoints = int(sample_rate * clip_edge)
    return seg[clip_edge_datapoints:-clip_edge_datapoints]

def load_folder(path:str, ifos:list[str]):
    '''
    load the glitch times and data associated with a "save" folder
    '''
    #if len(os.listdir(path)) != 6: #check that all files are in place
    #    return None

    if path[-1] == "/": path = path[:-1] #hopefully there aren't two...
    folder_name = path.split("/")[-1]
    assert len(folder_name.split("_")) == 2
    start = int(folder_name.split("_")[0])
    end = int(folder_name.split("_")[1])
    #print("start", start)

    for ifo in ifos:
        h5_file = find_h5(f"{path}/{ifo}/triggers/{ifo}:DCS-CALIB_STRAIN_CLEAN_C01/")
        if h5_file == None: return None

    loaded_data = dict()
    for ifo in ifos:
        #get the glitch times first
        h5_file = find_h5(f"{path}/{ifo}/triggers/{ifo}:DCS-CALIB_STRAIN_CLEAN_C01/")
        if h5_file == None: return None

        with h5py.File(h5_file, "r") as f:
            print('loaded data from h5', h5_file)
            triggers = f['triggers'][:]

        with h5py.File(f"{path}/detec_data_{ifo}.h5", "r") as f:
            X = f['ts'][:]

        #some statistics on the data
        data_statistics=0
        if data_statistics:
            print(f"start: {start}, end: {end}")
            print(f"duration, seconds: {end-start}")
            print(f"data length: {len(X)}")
            print(f"with data sampled at {4*4096}, len(data)/duration= {len(X)/4/4096}")


        sample_rate = 4*4096
        resample_rate = 4096 #don't need so many samples

        data = TimeSeries(X, sample_rate = sample_rate, t0 = start)
        if data_statistics:
            print(f"after creating time series, len(data) = {len(data)}")
            before_resample = len(data)
        
         
        if sample_rate != resample_rate:
            data = data.resample(resample_rate)

        if data_statistics:
            print(f"after resampling, len(data) = {len(data)}")
            print(f"ratio before to after: {before_resample/len(data)}")

        fftlength=1
        loaded_data[ifo] = {"triggers": triggers, 
                            "data":data, 
                            "asd":data.asd(fftlength=fftlength,overlap=0, method='welch', window='hanning')}

    return loaded_data

def whiten_bkg(bkg_seg_full:np.ndarray, sample_rate:int, H1_asd, L1_asd, clip_edge=1):
    #clip_edge=1
    ASDs = {"H1":H1_asd, 
                "L1":L1_asd}
    white_segX = []
    #print("into whiten, full:", bkg_segs_full.shape)
    for i, ifo in enumerate(["H1", "L1"]):
        bkg_seg = bkg_seg_full[i]
        print("in whiten, ", bkg_seg.shape)
        final_shape = (bkg_seg.shape[0]-2*int(clip_edge*sample_rate))
        print("final shape", final_shape)
        white_seg = np.zeros(final_shape)
        #for i, bkg_seg in enumerate(bkg_segs):
        white_seg = TimeSeries(bkg_seg, sample_rate=sample_rate).whiten(asd=ASDs[ifo])
        #white_seg = TimeSeries(bkg_seg, sample_rate=sample_rate).whiten()
        white_segX.append(clipping(white_seg, sample_rate, clip_edge=clip_edge))
        #all_white_segs.append(white_segs)
    #print("FINAL WHITE SEG", )
    #have to do clipping because of edge effects when whitening? check this...yes! have to
    FINAL_WHITEN = np.stack(white_segX)
    print("FINAL WHITEN", FINAL_WHITEN.shape)
    return FINAL_WHITEN

def clean_GW_events(event_times, data, fs, ts, tend):
    print("shapes into clean_GW_events, event_times, data", event_times.shape, data.shape)
    convert_index = lambda t: int(fs * (t-ts))
    bad_times = []
    for et in event_times:
        if et > ts+5 and et < tend-5:
            bad_times.append( convert_index(et))
    
    print("problematic times with GWs:", bad_times)
    clean_window = int(5*fs) #seconds
    if len(bad_times) == 0:
        return data[clean_window:-clean_window]

    #just cut off the edge instead of dealing with BBH's there
    sliced_data = np.zeros(shape=( int(data.shape[0]-clean_window*(bad_times) ), 2) )

    start = 0
    stop = 0
    marker = 0
    for time in bad_times:
        stop = int(time - clean_window//2)
        seg_len = stop - start
        sliced_data[marker:marker+seg_len, :] = data[start:stop, :]
        marker += seg_len
        start = int(time + clean_window//2)

    #return, while chopping off the first and last 5 seconds
    return sliced_data[clean_window:-clean_window, :]

def timeslide(data, fs):
    timeslide_step = 2 * fs
    step = timeslide_step
    n_slides = 10 # could have more, maybe manually increase later
    width = 8*fs
    n_samp = int(len(data)/width) #will round down, important!

    all_slides = np.empty(shape=(n_slides*n_samp, width, 2))
    for i in range(1, n_slides+1):
        slid = np.copy(data)

        #sliding the second detector
        slid[i*step:, 1] = data[:-i*step, 1]
        slid[:i*step, 1] = data[-i*step:, 1]

        #slicing the data up into samples and putting it into all slides
        all_slides[(i-1)*n_samp:i*n_samp]=slid[:n_samp*width].reshape(n_samp, width, 2)

    return all_slides

def main_timeslides(savedir, folder_path):
    try:
        os.makedirs(savedir)
    except FileExistsError:
        None

    ifos = ["H1", "L1"]
    loaded_data = load_folder(folder_path, ifos)
    if loaded_data == None:
        return None
    fs = 4096
    data = np.vstack([loaded_data["H1"]['data'], loaded_data["L1"]['data']])

    clip_edge_whiten=1
    whitened = whiten_bkg(data, fs, loaded_data["H1"]["asd"], loaded_data["L1"]["asd"], clip_edge_whiten)
    whitened = np.swapaxes(whitened, 0, 1)
    event_times = np.load("/home/ryan.raikman/s22/LIGO_EVENT_TIMES.npy")
    ts = int(folder_path.split("/")[-1].split("_")[0])
    tend = int(folder_path.split("/")[-1].split("_")[0])
    print("139 debug, folder, ts, tend", folder_path, ts, tend)

    ts += clip_edge_whiten*fs
    tend -= clip_edge_whiten*fs

    data_sliced = clean_GW_events(event_times, whitened, fs, ts, tend)

    data_timeslides = timeslide(data_sliced, fs)

    B_to_GB = 1e9
    print("size in GB", data_timeslides.size * data_timeslides.itemsize/B_to_GB)
    np.save(f"{savedir}/timeslide_data.npy", data_timeslides)

    
