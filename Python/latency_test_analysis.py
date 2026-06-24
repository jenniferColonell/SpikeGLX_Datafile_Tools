# -*- coding: utf-8 -*-
"""
Requires python 3

Analyze SGLX binary from of standard latency test.
The neural channels must contain the square wave (either LFP or unfiltered AP)
Neural channel to analyze and the threshold in mV must be set (lines 35, 39)

Requires access to functions in DemoReadSGLX module to interpret metadata.

"""
import numpy as np
import os 
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import Tk
from tkinter import filedialog
from DemoReadSGLXData import readSGLX

def main():

    # Get file from user
    root = Tk()         # create the Tkinter widget
    root.withdraw()     # hide the Tkinter root window

    # Windows specific; forces the window to appear in front
    root.attributes("-topmost", True)

    binFullPath = Path(filedialog.askopenfilename(title="Select binary file"))
    root.destroy()      # destroy the Tkinter widget
    
    outDir = binFullPath.parent
    
    # neural channel to analyze, zero based 
    chan = [5]
    
    # threshold in uV. Should match threshold used in latency test code
    # Note that threshold in CPP latency test code not converted to uV
    th = 500
    
    perSec = 0.1  # period of square wave sent to buffer
    
    # read the single digital word in the imec data file == sync channel
    dw = 0

    # Zero-based Line index of the line to read
    # SYNC/SMA input = line 6
    smaLine = [6]

    # Read in metadata; returns a dictionary with string for values
    meta = readSGLX.readMeta(binFullPath)
    
    # parameters for the data
    sRate = readSGLX.SampRate(meta)
    fileSizeBytes = int(meta['fileSizeBytes'])
    nSavedChans = int(meta['nSavedChans'])
    nFileSamp = fileSizeBytes/(2*nSavedChans)   
    origChans = readSGLX.OriginalChans(meta)
    savedInd = np.where(origChans == chan)[0]
    if savedInd.size != 1:
        print('analysis channel {chan} not in saved set')
        return
    
    
    # batch size < half period, so there is always one or zero edges in a batch
    batchSamp = int(np.floor(0.4*perSec*sRate))
    currSamp = 0
      
    rawData = readSGLX.makeMemMapRaw(binFullPath, meta)
    
    numEdges = 0
    maxEdges = 1000
    
    measLatencyMS = np.zeros((maxEdges,))
    
    # loop over all the data in 0.4*period batches
    # If a batch contains positive going edges in both the neural channel and
    # the digital signal containing hte response, measure the difference 
    # between the two. If the batch does not contain both edges, that edge 
    # is skipped.
    while (currSamp < nFileSamp - batchSamp) and (numEdges < maxEdges) :
        # print('currSamp: ' + repr(currSamp))
        selectData = rawData[savedInd, currSamp:currSamp+batchSamp]
        convData = np.squeeze(1e6*readSGLX.GainCorrectIM(selectData, savedInd, meta)) # convData in uV

        rspData = np.squeeze(readSGLX.ExtractDigital(rawData, currSamp, currSamp+batchSamp-1, dw,
                                 smaLine, meta))

        if currSamp == 0:
            np.save('neuralData.npy', convData);
            np.save('rspData.npy', rspData)
            
        # array of times for plot
        tDat = np.arange(currSamp, currSamp+batchSamp)
        tDat = 1000*tDat/sRate      # plot time axis in msec
        
        # #plotting for diagnostics
        # # Plot the neural channel
        # fig, ax = plt.subplots()
        # ax.plot(tDat, convData)
        # plt.show()

        # # Plot the digital channel
        # fig, ax = plt.subplots()
        # ax.plot(tDat, rspData)
        # plt.show()           

        # is there a postive going edge in the neural channel
        # get highest index of below threshold data points
        convAboveTh = np.nonzero(convData > th)
        convBelowTh = np.nonzero(convData < th)
        if convAboveTh[0].size > 0 and convBelowTh[0].size > 0 :
            # there's a transition that could be a postive edge
            maxBelowTh = np.max(convBelowTh)
            minAboveTh = np.min(convAboveTh)
            # postive edge has minAbove just a little larger than maxBelow
            diff = minAboveTh - maxBelowTh
            # print('diff: ' + repr(diff))
            if (diff >= 0) and (diff < 20):
                # positive edge detected at minAboveTh
                # serach forward from this point for the matched edge in 
                rspEdge = rspData[minAboveTh:]
                rspEdgeAboveZero = np.nonzero(rspEdge > 0)               
                if rspEdge[0] == 0 and rspEdgeAboveZero[0].size > 0 : # a real edge should start w/ rsp = 0                    
                    currLatMS = 1000*float(np.min(rspEdgeAboveZero))/sRate
                    if currLatMS > 0 and currLatMS < 200:
                        # successful measurment
                        # print(currLatMS)
                        measLatencyMS[numEdges] = currLatMS
                        numEdges = numEdges + 1
                    
        currSamp = currSamp + batchSamp
        
    measLatencyMS = measLatencyMS[0:numEdges]
    np.save(os.path.join(outDir,'measLatency.npy'),measLatencyMS)
        
    bins = np.arange(1,5,0.1)
    plt.hist(measLatencyMS, bins)
    plt.xlabel("Measured latency (ms)")
    
    meanLatency = np.mean(measLatencyMS)
    stdLatency = np.std(measLatencyMS)
    maxLatency = np.max(measLatencyMS)
    pc75 = np.percentile(measLatencyMS,75)
    pc50 = np.percentile(measLatencyMS,50)
    print( f'Mean, standard deviation, and max latency in ms: {meanLatency:.1f}, {stdLatency:.1f}, {maxLatency:.1f}')
    print( f'50th percenti;e, 75th percentile: {pc50:.1f}, {pc75:.1f} ')    


if __name__ == "__main__":
    main()
