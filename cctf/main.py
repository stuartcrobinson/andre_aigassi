import cctf.cctfTools as ctools

# ctools.generateCctfVideoFileFromVideo()
# ctools.generateCctfTrackNetBadmintonImagesNpyFile(224)
# ctools.generateCctfTrackNetBadmintonImagesNpyFile(112)

# 1.  do qa on tracknet badminton training data
ctools.runQaOnTracknetTrainingDataPairsOnly(50, startFrame=605)#, endFrame=None)
# ctools.runQaOnTracknetTrainingData(900, startFrame=600)#, endFrame=None)
# ctools.generateCctfTrackNetBadmintonImagesNpyFile(900, startFrame=170, endFrame=2220, brightness=4)



'''
okay.  tracknet training data is not so hot.

the tracking information is fine, but i think the camera is bad

every 6 frames or so, the frame is duplicated.
also sometimes there are blips where it's just crazy.

we need to clean up the training data.
need: make a list of all bad frames.  skip those during everything.

how to detect bad frames?  
bad: max value < 50 OR ave value > 1

we need to CUT the video stream wherever the mean frameDiff is > 1. this is when camera was stopped and started i think.
so at these cuts, act like the video is just starting.  

so have two types of frames labels.

1.  IGNORE:  

this is for frames that are nearly identical to previous frame.
ignore them when generating cctf images.  as tho they never existed
initialize output .npy-bound array to size of NUM_FRAMES minus NUM_IGNORE

2.  CUT:

this is for frames that are drastically different from prior frame.
here, we should start over.  set all prevN frames to equal curr frame.  like in the beginning.
 

TODO - scroll through images.  make json object with two arrays: ignore and cut

for h = 50

ignore if max < 20
cut if mean > 4 ??

/usr/local/bin/python3.6 /Users/stuartrobinson/repos/computervision/andre_aigassi/cctf/main.py
default im shape: (720, 1280)
targetHeight: 900
frame: 860  max: 212   mean: 5.295 sum: 4288958 normdiff: 833.636
frame: 861  max: 166   mean: 0.663 sum:  537054 normdiff: 5.360
frame: 862  max: 174   mean: 0.668 sum:  541103 normdiff: 14.243
frame: 863  max: 184   mean: 0.675 sum:  546615 normdiff: 21.720
frame: 864  max: 19    mean: 0.085 sum:   68695 normdiff: 7.345
frame: 865  max: 176   mean: 0.635 sum:  514435 normdiff: 29.027
frame: 866  max: 165   mean: 0.652 sum:  528076 normdiff: 8.949
frame: 867  max: 170   mean: 0.651 sum:  526991 normdiff: 2.245
frame: 868  max: 161   mean: 0.644 sum:  521906 normdiff: 41.149
frame: 869  max: 168   mean: 0.601 sum:  486904 normdiff: 43.065
frame: 870  max: 19    mean: 0.045 sum:   36093 normdiff: 6.021
frame: 871  max: 168   mean: 0.788 sum:  638595 normdiff: 20.442
frame: 872  max: 158   mean: 0.586 sum:  474970 normdiff: 13.511
frame: 873  max: 158   mean: 0.637 sum:  515919 normdiff: 11.918
frame: 874  max: 129   mean: 0.586 sum:  474630 normdiff: 15.687
frame: 875  max: 149   mean: 0.546 sum:  442129 normdiff: 9.812
frame: 876  max: 17    mean: 0.039 sum:   31501 normdiff: 0.194
frame: 877  max: 161   mean: 0.520 sum:  420889 normdiff: 26.890
frame: 878  max: 171   mean: 0.548 sum:  444238 normdiff: 9.943
frame: 879  max: 13    mean: 0.016 sum:   13290 normdiff: 1.601
frame: 880  max: 182   mean: 2.038 sum: 1650815 normdiff: 86.362
frame: 881  max: 172   mean: 0.543 sum:  439498 normdiff: 12.080
frame: 882  max: 14    mean: 0.034 sum:   27635 normdiff: 6.188
frame: 883  max: 151   mean: 0.567 sum:  459581 normdiff: 5.421
frame: 884  max: 154   mean: 0.560 sum:  453479 normdiff: 7.383
frame: 885  max: 159   mean: 0.527 sum:  426981 normdiff: 44.573
frame: 886  max: 137   mean: 0.538 sum:  435696 normdiff: 43.576
frame: 887  max: 15    mean: 0.066 sum:   53110 normdiff: 4.001
frame: 888  max: 137   mean: 0.483 sum:  391602 normdiff: 5.647

Process finished with exit code 0

what about 

a crazy blip
                           :  0.4373283950617284
frame: 610  of  18241
            Max element from Numpy Array :  134
            ave:                                                  :  0.5811654320987655
frame: 611  of  18241
            Max element from Numpy Array :  19
            ave:                                                  :  0.04911234567901235
frame: 612  of  18241
            Max element from Numpy Array :  213
            ave:                                                  :  5.290769135802469
frame: 613  of  18241
            Max element from Numpy Array :  26
            ave:                                                  :  0.07085308641975309
frame: 614  of  18241
            Max element from Numpy Array :  21
            ave:                                                  :  0.0642283950617284
frame: 615  of  18241
            Max element from Numpy Array :  24
            ave:                                                  :  0.06680617283950617
frame: 616  of  18241
            Max element from Numpy Array :  24 

periodic near-duplications:

           Max element from Numpy Array :  143
            ave:                                                  :  0.2698456790123457
frame: 667  of  18241
            Max element from Numpy Array :  12
            ave:                                                  :  0.0721641975308642
frame: 668  of  18241
            Max element from Numpy Array :  129
            ave:                                                  :  0.5968345679012346
frame: 669  of  18241
            Max element from Numpy Array :  124
            ave:                                                  :  0.3553222222222222
frame: 670  of  18241
            Max element from Numpy Array :  134
            ave:                                                  :  0.3541814814814815
frame: 671  of  18241
            Max element from Numpy Array :  136
            ave:                                                  :  0.39615185185185187
frame: 672  of  18241
            Max element from Numpy Array :  134
            ave:                                                  :  0.38307654320987655
frame: 673  of  18241
            Max element from Numpy Array :  14
            ave:                                                  :  0.05031604938271605
frame: 674  of  18241
            Max element from Numpy Array :  151
            ave:                                                  :  0.3887654320987654
frame: 675  of  18241
            Max element from Numpy Array :  158
            ave:                                                  :  0.416958024691358
frame: 676  of  18241
            Max element from Numpy Array :  160
            ave:                                                  :  0.4516432098765432
frame: 677  of  18241
            Max element from Numpy Array :  151
            ave:                                                  :  0.442820987654321
frame: 678  of  18241
            Max element from Numpy Array :  146
            ave:                                                  :  0.42384074074074074
frame: 679  of  18241
            Max element from Numpy Array :  15
            ave:                                                  :  0.06089876543209877
frame: 680  of  18241
            Max element from Numpy Array :  110

'''