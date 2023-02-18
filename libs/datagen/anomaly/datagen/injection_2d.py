import os
import h5py
import bilby
import numpy as np
import scipy.signal as sig
from scipy.stats import cosine as cosine_distribution
from gwpy.timeseries import TimeSeries
from typing import Callable
from anomaly.datagen.waveforms import olib_time_domain_sine_gaussian

'''
some code taken from Ethan Marx
https://git.ligo.org/olib/olib/-/blob/main/libs/injection/olib/injection/injection.py
'''
def find_h5(path):
    h5_file = None
    if not  os.path.exists(path): return None
    for file in os.listdir(path):
        if file[-3:] == ".h5":
            assert h5_file is None #make sure only 1 h5 file 
            h5_file = path + "/" + file

    assert h5_file is not None #did not find h5 file
    return h5_file

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

def get_loud_segments(ifo_data:dict, N:int, segment_length:float):
    '''
    sort the glitches by SNR, and return the N loudest ones
    If there are < N glitches, return the number of glitches, i.e. will not crash
    '''
    glitch_snr_bar = 10

    glitch_times = ifo_data["triggers"]['time']
    t0 = ifo_data['data'].t0.value
    tend = t0 + len(ifo_data['data'])/ifo_data['data'].sample_rate.value

    glitch_snrs = ifo_data["triggers"]['snr']
    sort_by_snr = glitch_snrs.argsort()[::-1] #create a sorting, in descending order
    #print(glitch_snrs[sort_by_snr])
    glitch_times_sorted = glitch_times[sort_by_snr]

    glitch_start_times = []
    for i in range(min(len(glitch_times_sorted), N)):
        #going to set a glitch SNR bar because the really quiet ones don't really do much
        if glitch_snrs[sort_by_snr][i] < glitch_snr_bar:
            continue
        #want times for the beginning of the segment, and lets have the glitch centered
        gt = glitch_times_sorted[i]
        glitch_start_times.append(gt-segment_length/2)

    return glitch_start_times

def get_quiet_segments(ifo_data:dict, N:int, segment_length:float):
    '''
    get N times that are away from the glitches
    '''
    edge = segment_length

    glitch_times = ifo_data["triggers"]['time']
    t0 = ifo_data['data'].t0.value
    tend = t0 + len(ifo_data['data'])/ifo_data['data'].sample_rate.value
    
    valid_start_times = np.arange(t0, tend-segment_length, segment_length)
    open_times = np.ones(valid_start_times.shape)

    #print("t0", t0)
    #print("valid_start_times", valid_start_times)
    for gt in glitch_times:
        #print("searching", gt, "into", valid_start_times)
        idx = np.searchsorted(valid_start_times, gt)
        #print("got result", idx)
        #print([float(elem) for elem in valid_start_times[idx-1:idx+2]], gt)
        #get rid of 3 indicies
        for i in range(idx-2, idx+2):
           #print(idx)
            try:
                open_times[i] = 0
            except IndexError:
                #print("something happened out of bounds with", idx)
                None #this is dangerous, just using it for now to deal with glitches on the edge

    total_available = np.sum(open_times)
    #assert total_available >= N #if this fails, then need to revise choosing strategy
    if total_available < N: #manually setting the maximuim
        N = int(total_available)

    #convert to bool mask
    open_times = (open_times != 0)
    valid_start_times = valid_start_times[open_times]

    #now just take from valid start times without replacement

    return valid_start_times
    quiet_times = np.random.choice(valid_start_times, N, replace=False)

    #one last check
    for elem in quiet_times:

        assert np.abs(glitch_times-elem).min() >= segment_length
        assert np.abs(glitch_times-(elem+segment_length)).min() >= segment_length
    return quiet_times
    
def slice_bkg_segments(ifo_data:dict, data, start_times:list[float], segment_length:float):
    #print("got for start times,", start_times)
    #turn the start times into background segments
    fs = ifo_data['data'].sample_rate.value
    t0 = ifo_data['data'].t0.value
    N_datapoints = int(segment_length * fs)
    bkg_segs = np.zeros(shape=(2, len(start_times), N_datapoints))
    bkg_timeseries = []

    for i in range(len(start_times)):
        slx = slice( int((start_times[i]-t0)*fs),  int((start_times[i]-t0)*fs)+N_datapoints)
        slice_segment = data[:, slx]
        #print("WHERE THE ASSERTION WAS FAILING", slice_segment.t0.value, start_times[i])
        #assert np.isclose(slice_segment.t0.value, start_times[i]) #check that timing lines up
        toggle_noise=1
        bkg_segs[:, i] = np.array(slice_segment) * toggle_noise
        bkg_timeseries.append(slice_segment)

    return bkg_segs, bkg_timeseries

def inject_hplus_hcross(bkg_seg:np.ndarray,
                        pols:dict,
                        sample_rate:int,
                        segment_length:int):

    final_injects = []
    #sample ra, dec, geocent_time, psi for the signal
    ra = np.random.uniform(0, 2*np.pi)
    psi = np.random.uniform(0, 2*np.pi)
    dec = cosine_distribution.rvs(size=1)[0]
    geocent_time = segment_length/2#put it in the middle
    #print("dec", dec)

    for i, ifo in enumerate(["H1", "L1"]):
        bkgX = bkg_seg[i]# get rid of noise? to see if signal is showing up
        print("bkgX", bkgX.shape)
        print("background data", bkgX)
        bilby_ifo = bilby.gw.detector.networks.get_empty_interferometer(
                    ifo
                )
        full_seg = TimeSeries(bkgX, sample_rate=sample_rate, t0 = 0)
        bilby_ifo.strain_data.set_from_gwpy_timeseries(full_seg)

        #psi = 0 #sample uniform 0 -> 2pi?
        
        injection = np.zeros(len(bkgX))
        for mode, polarization in pols.items(): #make sure that this is formatted correctly
            response = bilby_ifo.antenna_response( #this gives just a float, my interpretation is the
                ra, dec, geocent_time, psi, mode #inclination angle of the detector to the source
            )
            response *= 20 #to make SN signals show up
            midp = len(injection)//2
            if len(polarization) % 2 == 1:
                polarization=polarization[:-1]
            #betting on the fact that the length of polarization is even
            half_polar = len(polarization)//2
            #print(len(polarization))
            assert len(polarization) % 2 == 0  #off by one datapoint if this fails, don't think it matters
            slx = slice(midp-half_polar, midp+half_polar)
            injection[slx] += response * polarization

        signal = TimeSeries(injection, times = full_seg.times, unit=full_seg.unit)

        full_seg = full_seg.inject(signal)
        final_injects.append(full_seg)

    final_bothdetector = np.stack(final_injects)      
    #print("FINAL SHAPE", final_bothdetector.shape)
    return final_bothdetector

def inject_waveforms(waveform:Callable,
                    bkg_segs_2d:np.ndarray,
                    sample_rate:int,
                    t0:int,
                    segment_length:int,
                    prior_file,
                    #ifo:str,
                    waveform_arguments=None,
                    domain:str='time',
                    center_type:str=None):
    N_datapoints = segment_length * sample_rate

    if domain == 'time':
        waveform_generator = bilby.gw.WaveformGenerator(
                duration=segment_length,
                sampling_frequency=sample_rate,
                time_domain_source_model=waveform,
                waveform_arguments=waveform_arguments
            )
    else:
        assert domain in ["freq", "frequency"]
        waveform_generator = bilby.gw.WaveformGenerator(
                duration=segment_length,
                sampling_frequency=sample_rate,
                frequency_domain_source_model=waveform,
                waveform_arguments=waveform_arguments
            )

    #now sample params

    #support passing in either the name of the file or the actual prior
    if type(prior_file) == str:
        priors = bilby.gw.prior.PriorDict(prior_file)
    else:
        priors = prior_file
    injection_parameters = priors.sample(bkg_segs_2d.shape[1])
    #print("injection parameters,", injection_parameters)

    #reshaping, code taken from https://git.ligo.org/olib/olib/-/blob/main/libs/injection/olib/injection/injection.py
    #lns 107-110
    injection_parameters = [
            dict(zip(injection_parameters, col))
            for col in zip(*injection_parameters.values())
        ]

    final_injects = []
    for i, ifo in enumerate(["H1", "L1"]):
        bkg_segs = bkg_segs_2d[i]
        bilby_ifo = bilby.gw.detector.networks.get_empty_interferometer(
                    ifo
                )
        #make full seg, basically a segment with all the bkg_segs appended together
        print("BKG SEG SHAPE", bkg_segs.shape, "hstack", np.hstack(bkg_segs).shape)
        full_seg = TimeSeries(np.hstack(bkg_segs), sample_rate=sample_rate, t0 = 0)

        bilby_ifo.strain_data.set_from_gwpy_timeseries(full_seg)

        for i, p in enumerate(injection_parameters):
            #print("param set", p)
            p['luminosity_distance']=np.random.uniform(50, 200)
            ra = p['ra']
            dec = p['dec']
            start_time = (i * segment_length)*sample_rate #start of the segment to be injected into
            geocent_time = t0 + start_time
            p['geocent_time'] = 0 * sample_rate * segment_length / 2 #center the injection in the middle of the signal
            psi = p['psi']
            #print("start time", start_time)
            #print('geocent_time', p['geocent_time'])

            #get hplus, hcross
            polarizations = waveform_generator.time_domain_strain(p)

            injection = np.zeros(len(full_seg))
            #print("full seg length", len(full_seg))
            # loop over polarizaitions applying detector response
            for mode, polarization in polarizations.items():
                #print("mode", mode)
                #print("polarization", polarization)
                #np.save("./polarization.npy", polarization)

                response = bilby_ifo.antenna_response( #this gives just a float, my interpretation is the
                    ra, dec, geocent_time, psi, mode #inclination angle of the detector to the source
                )

                #print(injection.shape, response.shape, polarization.shape)
                #approximate tricks for finding center and doing injection based on that
                if center_type == None: #just none for the sine gaussian models
                    centre = np.argmax(np.abs(polarization)) #this works for sine gaussian but can be problematic for other ones
                    #want center to be halfway through the segment, at least for now
                    a = start_time + N_datapoints//2 - centre
                    #truncate so it doesn't go over the edge, should be ok since all zeros
                    inj_slice = slice(a, a + centre*2)
                    #print(start_time, centre)
                    #all good here for BBH polarization
                    #print("centre", centre, "polar shape", polarization.shape)
                    #print("polarization shape", polarization.shape)
                    #np.save("./inj_polarization.npy", polarization)
                    injection[inj_slice] += response * polarization[:centre*2]
                elif center_type in ["BBH", "bbh"]:
                    #print("full seg", len(full_seg), "polar", len(polarization))
                    #print("A", segment_length*sample_rate )
                    #print("indexing injection in ", start_time,start_time+segment_length*sample_rate)
                    #print("response", response)
                    #injection[start_time:start_time+(segment_length-2)*sample_rate] += response * polarization[-((segment_length-2)*sample_rate):]

                    #better idea, put the end of the BBH at the middle? that way clipping isn'ty an issue
                    injection[start_time:start_time+segment_length*sample_rate//2] += response*polarization[-(segment_length*sample_rate//2):]

            signal = TimeSeries(injection, times = full_seg.times, unit=full_seg.unit)
            #maybe need to do the shifting stuff here
            full_seg = full_seg.inject(signal)

        #now need to chop back up the full_seg
        injected_bkgs = np.array(full_seg).reshape(bkg_segs.shape)
        final_injects.append(injected_bkgs)
    

    final_bothdetector = np.stack(final_injects)
    print("FINAL INJECT SHAPE", final_bothdetector.shape)
    
    return final_bothdetector

def clipping(seg:TimeSeries, sample_rate:int, clip_edge:int=1):
    clip_edge_datapoints = int(sample_rate * clip_edge)
    return seg[clip_edge_datapoints:-clip_edge_datapoints]

def whiten_bkgs(bkg_segs_full:np.ndarray, sample_rate:int, H1_asd, L1_asd):
    clip_edge=1
    ASDs = {"H1":H1_asd, 
                "L1":L1_asd}
    all_white_segs = []
    print("into whiten, full:", bkg_segs_full.shape)
    for i, ifo in enumerate(["H1", "L1"]):
        bkg_segs = bkg_segs_full[i]
        print("in whiten, ", bkg_segs.shape)
        final_shape = (bkg_segs.shape[0], bkg_segs.shape[1]-2*int(clip_edge*sample_rate))
        white_segs = np.zeros(final_shape)
        for i, bkg_seg in enumerate(bkg_segs):
            white_seg = TimeSeries(bkg_seg, sample_rate=sample_rate).whiten(asd=ASDs[ifo])
            #white_seg = TimeSeries(bkg_seg, sample_rate=sample_rate).whiten()
            white_segs[i] = clipping(white_seg, sample_rate, clip_edge=clip_edge)
        all_white_segs.append(white_segs)

    #have to do clipping because of edge effects when whitening? check this...yes! have to
    FINAL_WHITEN = np.stack(all_white_segs)
    print("FINAL WHITEN", FINAL_WHITEN.shape)
    return FINAL_WHITEN

def whiten_bkgs_singular(bkg_seg_full:np.ndarray, sample_rate:int, H1_asd, L1_asd):
    clip_edge=1
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

def get_bkg_segs(loaded_data, data, N, segment_length): 
#note - by here, N samples have NOT been drawn
    quiet_times_H1 = get_quiet_segments(loaded_data["H1"], N, segment_length) 
    quiet_times_L1 = get_quiet_segments(loaded_data["L1"], N, segment_length)

    quiet_times = np.intersect1d(quiet_times_H1, quiet_times_L1)
    #print("after intersection", quiet_times)
    N = min(N, len(quiet_times))
    quiet_times = np.random.choice(quiet_times, N, replace=False)
    
   # print("quiet times, ", quiet_times)
    
    #assert False 
    #passing loaded_data here for reference to values like t0 and fs
    bkg_segs, _ = slice_bkg_segments(loaded_data["H1"], data, quiet_times, 
                                    segment_length)
    return bkg_segs

def main_SNGW(savedir, folder_path, polarization_files):
    print("ENTERING main_SNGW")
    try: 
        os.makedirs(savedir)
    except FileExistsError:
        None

    ifos=["H1", "L1"]
    loaded_data = load_folder(folder_path, ifos)
    data = np.vstack([loaded_data["H1"]['data'], loaded_data["L1"]['data']])

    fs = 4096
    segment_length = 4 #seconds, confirm that the length of SN signals is less than this

    N = len(os.listdir(polarization_files))
    N = 25 #only doing 5 samples from each folder
    #print("N", N)
    bkg_segs = get_bkg_segs(loaded_data, data, N, segment_length)

    #print("BKG SEGS", bkg_segs.shape)

    #load up the polarization files
    
    #this should work just fine with a pair of files
    if 0:
        angle_values = set()
        for polarization_file in os.listdir(polarization_files):
            #print(polarization_file, pick_out_angles(polarization_file))
            angle_values.add(pick_out_angles(polarization_file))

    if 0: #this is working for a particular set of waveforms (Powell), s18_3d
        polarizations = []
        for phi, theta in angle_values:
            hcross = f"s18_3d_phi{phi}_theta{theta}_16384Hz_hcross.txt"
            hplus = f"s18_3d_phi{phi}_theta{theta}_16384Hz_hplus.txt"
            #print("phi, theta", phi, theta)
            hcross = np.loadtxt(polarization_files + hcross)
            hplus = np.loadtxt(polarization_files + hplus)

            new_len = int(len(hcross) * 4096/16384) #make sure that inputted SN signals are 16384 Hz
            hcross = sig.resample(hcross, new_len)
            hplus = sig.resample(hplus, new_len)
            #print("polarization shapes")
            #print(hcross.shape)
            #print(hplus.shape)

            #have to do downsampling from 16384 hz to 4096 hz
            polarizations.append({"cross": hcross, "plus":hplus})
        # print("sucsessfully loaded txts")

    N_SN_samples = 25
    if 1:#this is for the format of each pair being saved under a particular folder
        polarizations = []
        names = []
        N_SN_samples = min(N_SN_samples, len(os.listdir(polarization_files)))
        for name in np.random.choice(os.listdir(polarization_files), size=N_SN_samples, replace=False) :
            hcross, hplus = sorted(os.listdir(polarization_files + f"/{name}/")) #perhaps not a great way to do it

            hcross = np.loadtxt(f"{polarization_files}/{name}/{hcross}")
            hplus = np.loadtxt(f"{polarization_files}/{name}/{hplus}")

            new_len = int(len(hcross) * 4096/16384) #make sure that inputted SN signals are 16384 Hz
            hcross = sig.resample(hcross, new_len)
            hplus = sig.resample(hplus, new_len)
            #print("polarization shapes")
            #print(hcross.shape)
            #print(hplus.shape)

            #have to do downsampling from 16384 hz to 4096 hz
            polarizations.append({"cross": hcross, "plus":hplus})
            names.append(name)
    
    for i, pols in enumerate(polarizations):
        #print("bkg seg shape",bkg_segs[i].shape)
        #print("DEBUG 454", i, pols, bkg_segs.shape[1])
        if i >= bkg_segs.shape[1]: #didn't generate enough bkg samples, this is generally fine unless small overall samples
            break 
        injected_waveform = inject_hplus_hcross(bkg_segs[:, i, :],
                        pols,
                        4096,
                        segment_length)
        print("before whitening", injected_waveform.shape)
        whitened_segs = whiten_bkgs_singular(injected_waveform, fs, loaded_data["H1"]["asd"], loaded_data["L1"]["asd"])
        np.save(f"{savedir}/{names[i]}_SN.npy", np.array(whitened_segs))


def main_big_bkg_segs(savedir, folder_path=None, prior_file=None):
    ifos = ["H1", "L1"]
    loaded_data = load_folder(folder_path, ifos)
    fs = 4096
    data = np.vstack([loaded_data["H1"]['data'], loaded_data["L1"]['data']])

    print("517 data.shape", data.shape)


def main_all3(savedir, N=20, folder_path = None, prior_file = None):
    try:  #important that this happens at the top,
        #so multiple ones don't run on the same data file
        os.makedirs(savedir)
    except FileExistsError:
        None

    ifos = ["H1", "L1"]
    if folder_path == None: #long default argument
        folder_path = "/home/ryan.raikman/s22/anomaly/data2/glitches/1252150173_1252152348/"
    
    loaded_data = load_folder(folder_path, ifos)
    if loaded_data == None:
        print("aborting due to missing data streams")
        return None

    #default argument
    if prior_file == None:
        prior_file = "/home/ryan.raikman/s22/forks/gw-anomaly/libs/datagen/anomaly/datagen/prior.prior"

    fs = 4096
    segment_length = 5 #seconds
    #for ifo in ifos:

    #stack the loaded data into 2-d np array (2, N)
    data = np.vstack([loaded_data["H1"]['data'], loaded_data["L1"]['data']])

    do_BBH = False
    do_BKG = False
    do_SG = False
    do_GLITCH = True 

    if do_BBH:
        #INJECTION
        assert loaded_data["H1"]['data'].t0.value == loaded_data["L1"]['data'].t0.value
        bkg_segs = get_bkg_segs(loaded_data, data, N, segment_length)
        
        #BNS: IMRPhenomDNRTidal_v2 [50] Approximant.
        BBH_waveform_args = dict(waveform_approximant='IMRPhenomPv2',
                            reference_frequency=50., minimum_frequency=20.)
        #BBH injection
        injected_segs = inject_waveforms(bilby.gw.source.lal_binary_black_hole,
            bkg_segs, fs, loaded_data["H1"]['data'].t0.value ,segment_length, 
            prior_file=bilby.gw.prior.BBHPriorDict(),
            waveform_arguments = BBH_waveform_args, domain="freq",
            center_type="BBH")

        #assert False
        #need both ASDs here
        whitened_segs = whiten_bkgs(injected_segs, fs, loaded_data["H1"]["asd"], loaded_data["L1"]["asd"])

        np.save(savedir + "/bbh_segs.npy", whitened_segs)

    if do_SG:
        #SG injection for now
        #get fresh bkg segs
        bkg_segs = get_bkg_segs(loaded_data, data, N, segment_length)
        injected_segs = inject_waveforms(olib_time_domain_sine_gaussian,
                bkg_segs, fs, loaded_data["H1"]['data'].t0.value ,segment_length, prior_file)

        whitened_segs = whiten_bkgs(injected_segs, fs, loaded_data["H1"]["asd"], loaded_data["L1"]["asd"])

        np.save(savedir + "/injected_segs.npy", whitened_segs)

    if do_BKG:
        #BACKGROUND
        bkg_segs = get_bkg_segs(loaded_data, data, N, segment_length)

        whitened_segs = whiten_bkgs(bkg_segs, fs, loaded_data["H1"]["asd"], loaded_data["L1"]["asd"])
        np.save(savedir + "/bkg_segs.npy", whitened_segs)

    if do_GLITCH:
        #GLITCHES
        loud_times_H1 = get_loud_segments(loaded_data["H1"], N, segment_length) 
        loud_times_L1 = get_loud_segments(loaded_data["H1"], N, segment_length)

        loud_times = np.union1d(loud_times_H1, loud_times_L1)

        glitch_segs, _ = slice_bkg_segments(loaded_data["H1"], data, loud_times, 
                                        segment_length)

        whitened_segs = whiten_bkgs(glitch_segs, fs, loaded_data["H1"]["asd"], loaded_data["L1"]["asd"])

        np.save(savedir + "/glitch_segs.npy", whitened_segs)

