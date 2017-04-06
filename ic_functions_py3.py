"""Identify transcription factor binding sites and analyze sequence architecture"""

# Import libraries
import os
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import re # regular expressions
import collections
import pandas as pd
from scipy import stats
import subprocess

# Change pd settings to display all dataframe content
pd.set_option('display.max_rows',None)

##########################################################
#Function(s)
##########################################################

def geneIDdictionary(filename):
    """Create dictionaries that convert between current gene symbol and FlyBase identifiers

    Arguments
    ----------
    filename : string
        path to FBgn <=> Annotation ID file (fbgn_annotation_ID_*.tsv) from FlyBase.

    Returns
    -------
    geneID : dictionary
        A dictionary that maps current and previous FlyBase identifiers (FBgn#s and CG#s)
        to the current gene symbol.
    NametoIDs : dictionary
        A dictionary that maps the current gene symbol to all possible (i.e. current and
        previous) FlyBase identifiers (FBgn#s and CG#s).
    NametoFBid : dictionary
        A dictionary that maps the current gene symbol to the current FBgn#.

    Notes
    -----
    Download the file from:
    FlyBase > Downloads > Current release > Genes > FBgn <=> Annotation ID

    Examples
    --------
    >>> geneID,NametoIDs,NametoFBid = \
                            geneIDdictionary('Datasets/fbgn_annotation_ID_fb_2016_04.tsv')
    >>> NametoIDs['rump]
    ['FBgn0267790',
    'CG9373',
    'FBgn0020921',
    'FBgn0260010',
    'FBgn0037701',
    'FBgn0043867']
    >>> geneID['FBgn0260010']
    ['rump']
    >>> NametoIDs['rump']
    'FBgn0267790'
    """
    regex = re.compile('^(D\w+\\\\)?([\w\(\)-:\'\[\]]+)\t(\w+)\t([\w,]+)?[\s\t]?(\w+)\t?([\w,\(\)\[\]]+)?', re.MULTILINE)
    # group(1) = species (if not Dmel,optional)
    # group(2) = gene symbol
    # group(3) = primary FBgn
    # group(4) = secondary FBgns (optional)
    # group(5) = annotation ID
    # group(6) = secondary annotation IDs (optional)

    geneID = {} # initialize geneID dictionary
    NametoIDs = {}
    NametoFBid = {}
    with open(filename) as f:
        for line in f:
            m = regex.search(line)
            if m:
                IDs = [] # list of all possible identifiers associated with gene
                if not m.group(1): # if Dmel
                    NametoFBid[m.group(2)] = m.group(3)
                    IDs = [m.group(2),m.group(3),m.group(5)]
                    if m.group(4):
                        IDs.extend(str(m.group(4)).split(','))
                    if m.group(6):
                        IDs.extend(str(m.group(6)).split(','))
                    NametoIDs[m.group(2)] = IDs
                for id in IDs:
                    if id in geneID:
                        geneID[id].append(m.group(2))
                    else:
                        geneID[id] = [m.group(2)]
#             else: # sanity check
#               print line

    return geneID, NametoIDs, NametoFBid

def readFASTA(filename,form,orientation,geneID,regexKey,nameasis=0,fullname=1,nonmatch=0,sep=','):
    """Read FASTA file into dictionary

    Arguments
    ----------
    filename : string
        path to FASTA file of interest
    form : string
        'PWM' if the FASTA file contains position weight matrices (PWMs)
        'sequences' if the FastA file contains sequences
        'list_num' if the FastA file contains a list of numbers for each identifier
        'list_str' if the FastA file contains a list of strings for each identified
    orientation : string
        'h' if PWM is in a horizontal orientation (rows: bases, columns: positions)
        'v' if PWM is in a vertical orientation (rows: positions in motif, columns: bases)
    geneID : dictionary, default = geneID
        dictionary that maps current and previous FlyBase identifiers (FBgn#s and CG#s)
        to the current gene symbol.
    regexKey : compiled regular expression, i.e. re.compile(pattern)
        used to parse the description/header line for the desired identifier for each item
        in the FASTA file
    nameasis : integer
        0 (default) to convert regular expression search results to gene symbol
        1 to use the regular expression search results directly
    fullname : integer
        1 (default) if the full header line of the FASTA file needs to be saved (useful for PWMs)
        0 otherwise
    nonmatch : string
        0 (default) if you don't want to print the header lines that do not match the regex
        'print' if you do want to print the header lines that fail to match the regex
    sep : string
        separator used to parse lines that contain lists
        ',' (default) 


    Returns
    -------
    dictionary : dictionary
        A dictionary that maps the desired identifier to the item
        For PWMs, dictionary maps transcription factor to PWM
        For sequences, dictionary maps sequence name to sequence
    checkFBgn : list
        A list of identifiers that were associated with multiple gene symbols
        These should be checked later to make sure they don't refer to obsolete genes, etc
    PWMnames : dictionary
        A dictionary that maps identifiers with the full header lines of the FASTA file
        This is useful when there are multiple items in the FASTA file that might map to
        the same in the dictionary, and you need to know which one you ended up saving. In
        this case of PWMs, this is useful because readFASTA saves the PWM created with the
        most sequences, and there's no way to know which on that is before you've parsed
        the file.

    Notes
    -----
    In the case of PWMs, readFASTA saves the PWM created with the most sequences.

    Examples
    --------
    >>> PWM,checkFBgn,PWMnames = readFASTA("Datasets/UmassPGFE_PWMfreq_PublicDatasetA_20150901.txt", \
                          'PWM','v',geneID,re.compile('(FBgn\d*)'))
    >>> PWM['cad']
    array([[  3.46002970e+02,   2.46002030e+02,   3.10002030e+02,
              5.18002970e+02],
           [  1.28002970e+02,   1.35002030e+02,   1.16002030e+02,
              1.04100297e+03],
           [  1.80002970e+02,   1.50020300e+01,   6.00203000e+00,
              1.21900297e+03],
           [  6.38002970e+02,   1.00020300e+01,   5.10020300e+01,
              7.21002970e+02],
           [  1.42000297e+03,   2.03000000e-03,   2.03000000e-03,
              2.97000000e-03],
           [  2.97000000e-03,   1.20002030e+02,   3.30020300e+01,
              1.26700297e+03],
           [  2.05002970e+02,   1.70020300e+01,   3.97002030e+02,
              8.01002970e+02],
           [  8.10002970e+02,   2.20020300e+01,   5.59002030e+02,
              2.90029700e+01]])
    >>> PWMnames['cad']
    'Cad_SOLEXA_FBgn0000251'
    >>> checkFBgn
    ['FBgn0050420', 'FBgn0051782']
    >>> geneID['FBgn0050420']
    ['Atf-2', 'CG44247']
    >>> AP_enhancers = readFASTA('Datasets/ap_enc_12.fa','sequences',0,geneID,re.compile('([\w\-]+)_mel'))
    >>> AP_DV_TFs = readFASTA('Datasets/Modified/AP_DV_TFs.txt','list_str',0,geneID,0)
    >>> AP_DV_TFs['DV']
    ['da', 'dl', 'brk', 'Mad', 'Med', 'shn', 'sna', 'twi', 'zen', 'zen2']
    >>> score_data = readFASTA('patserAnalysis/score_data.txt','list_num',0,geneID,0)
    >>> score_data['cad'][45:55]
    [6.19, 6.19, 6.19, 6.18, 5.91, 6.32, 5.91, 5.91, 5.91, 5.91]
    """

    # Initializing dictionaries and lists
    dictionary = {} # dictionary of PWMs or sequences
    tempdictionary = np.empty([0]) # temporary repository for PWM or sequence
    Keys = [] # list of keys (names of PWMs or sequences)
    tempPWMnames = [] # temporary repository for full PWM name
    PWMnames = {} # dictionary of full PWM names and the associated gene symbol that
                  # is saved to dictionary
    checkFBgn = [] # list of FlyBaseIDs that should be check due to multiple gene references
    save = 0 # whether the PWM or sequence should be saved

    # Open and parse the FASTA file
    with open(filename) as f:
        for line in f:
            if line.startswith('>'):
                # Save the last fully concatenated PWM or sequence to the dictionary
                if save == 1 and len(tempdictionary) > 0:
                    # if the PWM is horizontal, switch to vertical orientation
                    # if form is 'PWM' and orientation is 'h':
                    #     tempdictionary = np.transpose(tempdictionary)
                    # If (Case 1) dictionary is empty
                    #    (Case 2) the last key is NOT a key in dictionary
                    #    (Case 3) the last key is a key in the dictionary and is FlyReg
                    #             BUT this PWM is not from FlyReg
                    #    (Case 4) the last key is a key in the dictionary BUT this PWM uses
                    #             more sequences and is not FlyReg
                    #    (Case 5) the last key is a key in the dictionary and is FlyReg
                    #             BUT this PWM has more sequences and is also FlyReg
                    if ((not dictionary) or \
                    (Keys[-1] not in dictionary.keys()) or \
                    (form is 'PWM' and Keys[-1] in dictionary.keys() and \
                    'FlyReg' in PWMnames[Keys[-1]] and 'FlyReg' not in tempPWMnames[-1]) or \
                    (form is 'PWM' and Keys[-1] in dictionary.keys() and \
                    sum(tempdictionary[2,:]) > sum(dictionary[Keys[-1]][2,:]) and \
                    'FlyReg' not in tempPWMnames[-1]) or \
                    (form is 'PWM' and Keys[-1] in dictionary.keys() and \
                    'FlyReg' in PWMnames[Keys[-1]] and \
                    sum(tempdictionary[2,:]) > sum(dictionary[Keys[-1]][2,:]) and \
                    'FlyReg' in tempPWMnames[-1])):
                        if form is 'PWM' and orientation is 'h':
                            dictionary[Keys[-1]] = np.transpose(tempdictionary)
                        else:
                            dictionary[Keys[-1]] = tempdictionary
                        if form is 'PWM':
                            PWMnames[Keys[-1]] = tempPWMnames[-1]
                # Save a new key
                if regexKey is not 0:
                    m = regexKey.search(line)
                    # group(1) = name of PWM or sequence
                    if m:
                        if form is 'PWM':
                            if nameasis is 1:
                                Keys.append(m.group(1))
                            else:
                                if ((len(geneID[m.group(1)]) > 1) and (m.group(1) not in checkFBgn)):
                                    checkFBgn.append(m.group(1))
                                Keys.append(''.join(geneID[m.group(1)][0]))
                            tempPWMnames.append(line[1:-1])
                        else:
                            Keys.append(m.group(1))
                        save = 1
                    else:
                        if nonmatch == 'print':
                            print('Does not match regular expression: %s' %line.strip())
                        save = 0
                else:
                    Keys.append(line.strip()[1:])
                    tempPWMnames.append(line.strip()[1:])
                    save = 1
                tempdictionary = np.empty([0])
            else:
                if len(tempdictionary) > 0:
                    if form is 'PWM':
                        tempdictionary = np.vstack((tempdictionary, \
                                                    np.array([float(val) for val in \
                                                    line.split()])))
                    elif form is 'list_num':
                        tempdictionary += [float(x) for x in line.strip().split(sep)]
                    elif form is 'list_str':
                        tempdictionary += line.strip().split(sep)
                    else:
                        tempdictionary += line.strip()
                else:
                    if form is 'PWM':
                        tempdictionary = np.array([float(val) for val in \
                                            line.split()])
                    elif form is 'list_num':
                        tempdictionary = [float(x) for x in line.strip().split(sep)]
                    elif form is 'list_str':
                        tempdictionary = line.strip().split(sep)
                    else:
                        tempdictionary = line.strip()
    # if there is only one PWM or sequence in the file / for the last item in the file
    if len(dictionary) == 0 and len(tempdictionary) > 0 or save == 1:
        if form is 'PWM' and orientation is 'h':
            dictionary[Keys[-1]] = np.transpose(tempdictionary)
        else:
            dictionary[Keys[-1]] = tempdictionary
        if form is 'PWM':
            PWMnames[Keys[-1]] = tempPWMnames[-1]

    # if unnamed items are in dictionary, print error
    if '' in dictionary.keys():
        print(dictionary[''])
    if form is 'PWM':
        if fullname == 1:
            return dictionary,checkFBgn,PWMnames
        else:
            return dictionary,checkFBgn
    else:
        return dictionary

def PWMdictionary(PFM,pseudocount):
    """Convert probability frequency matrices (PFMs) into probability weight matrices (PWMs)

    Arguments
    ----------
    PFM : dictionary
        dictionary of probability frequency matrices
    pseudocount : float
        a pseudocount weighted by a priori probability of the corresponding base is
        added to each element


    Returns
    -------
    PWM : dictionary
        A dictionary that maps protein name to its position weight matrix.

    Examples
    --------
    >>> PWM = PWMdictionary(PFM,0.01)
    >>> PWM['bcd']
    array([[-0.89,  0.34, -0.48,  0.45],
       [-1.07, -2.68, -3.77,  1.09],
       [ 1.21, -9.29, -9.29, -9.29],
       [ 1.2 , -2.68, -9.29, -9.29],
       [-9.29, -3.09, -0.78,  1.11],
       [-4.16,  1.46, -2.68, -1.03],
       [-0.85,  1.11, -0.9 , -0.55]])
    """
    PWM = {}
    # Set the prior: weight the given pseudocount with the background frequencies
    q = [0.297, 0.203, 0.203, 0.297] # background frequencies for DNA based in intergenic region
                                     # the order of the bases is A C G T
    for k,v in PFM.items():
        temp_PWM = np.empty([0,4])
        for position in v:
            temp_PWM = np.vstack((temp_PWM,np.log((position+[pseudocount*x for x in q])/(sum(position)+pseudocount)/q)))
        PWM[k] = temp_PWM

    return PWM

def scoreCalc(sequence_dictionary, TF_list, PWM_dictionary, score_data, perc=25):
    """Calculate scores for the k-mers of given sequences against the desired position
    weight matrices

    Arguments
    ----------
    sequence_dictionary : dictionary
        dictionary of sequences
    TF_list : list
        list of transcription factors
    PWM_dictionary : dictionary
        dictionary of position weight matrices (PWMs)
    score_data : dictionary
        dictionary of scores of aligned sequences for TFs in PWM dictionary
    perc : float, default = 25
        percentile of the scored aligned sequences that will determine the score cutoff;
        note that a percentile of 25 means that 75% of the aligned sequences scores make
        the cut

    Returns
    -------
    output : pandas dataframe
        A dataframe listing the enhancer, tf, position of k-mer in the enhancer, and the score,
        filtered by the score cutoff determined by the user-chosen percentile

    Examples
    --------
    >>> output = scoreCalc(AP_enhancers,AP_DV_TFs['AP'], PWM, score_data)
    >>> output.head(5)
        enhancer    tf      position    score
    0   slp2-minus3     bcd     245     5.67
    1   slp2-minus3     bcd     314     7.63
    2   slp2-minus3     bcd     453     5.48
    3   slp2-minus3     bcd     1037    5.62
    4   slp2-minus3     bcd     1087C   5.51
    """

    bases = ['A','C','G','T']
    base2index = dict([reversed(x) for x in enumerate(bases)])
    complement = dict(zip(bases,bases[::-1]))
    score_cutoff = dict(zip(score_data.keys(),[np.percentile(x,perc) for x in score_data.values()]))

    output = pd.DataFrame(columns=['enhancer','tf','position','score'])
    with open('patser_output.txt','w') as f:
        for TF in TF_list:
            for seq_name, sequence in sequence_dictionary.items():
                for i in range(len(sequence)-PWM_dictionary[TF].shape[0]+1):
                    subsequence = sequence[i:i+PWM_dictionary[TF].shape[0]]
                    if 'N' in subsequence:
                        continue
                    temp_score = 0
                    for position, base in enumerate(subsequence):
                        temp_score = temp_score + PWM_dictionary[TF][position,base2index[base]]
                    if np.round(temp_score,2) > score_cutoff[TF]:
                        f.write('%s\t%s\t%s\t%.2f\n' %(seq_name.replace('-','_'), TF, i+1, temp_score))
                    subsequence_RC = ''.join([complement[x] for x in subsequence][::-1])
                    temp_score = 0
                    for position, base in enumerate(subsequence_RC):
                        temp_score = temp_score + PWM_dictionary[TF][position,base2index[base]]
                    if np.round(temp_score,2) > score_cutoff[TF]:
                        f.write('%s\t%s\t%s\t%.2f\n' %(seq_name.replace('-','_'), TF, str(i+1)+'C', temp_score))
    output = pd.read_csv('patser_output.txt',sep='\t',header=None)
    output.columns = ['enhancer','tf','position','score']
    output[['score']] = output[['score']].astype(float)

    return output

def ElemeNT_scoring(sequence_dictionary,category,motif_list,PFM_dictionary,score_cutoff,filepath):
    """Calculate ElemeNT scores for the k-mers of given sequences against the desired position
    weight matrices

    Arguments
    ----------
    sequence_dictionary : dictionary
        dictionary of sequences
    category : string
        name of set of promoters
    motif_list : list
        list of motifs
    PFM_dictionary : dictionary
        dictionary of position frequency matrices (PFMs)
    score_cutoff : dictionary
        dictionary of ElemeNT score cutoffs determined in Sloutskin, et al.
    filepath : string
        current working directory

    Returns
    -------
    output : pandas dataframe
        A dataframe listing the enhancer, tf, position of k-mer in the enhancer, and the score,
        filtered by the score cutoff determined by the user-chosen percentile

    Examples
    --------
    >>> output = ElemeNT_scoring(promoters,'promoters',pmotif_list,pPFM,ES_cutoff,os.getcwd())
    >>> output.head(5)
        sequence    motif   position    score
    0   ECSIT_1     BREd    71  0.6065
    1   Fhos_4  BREd    32  0.9317
    2   CG33552_1   BREd    58  0.5349
    3   CG33552_1   BREd    61  0.7859
    4   mrt_2   BREd    63  0.5518
    """

    bases = ['A','C','G','T']
    base2index = dict([reversed(x) for x in enumerate(bases)])
    
    output = pd.DataFrame(columns=['sequence','motif','position','score'])
    with open('%s/patserOut/ElemeNT_output_%s.txt' %(filepath,category),'w') as f:
        for motif in motif_list:
            for seq_name, sequence in sequence_dictionary.items():
                for i in range(len(sequence)-PFM_dictionary[motif].shape[0]+1):
                    subsequence = sequence[i:i+PFM_dictionary[motif].shape[0]]
                    if 'N' in subsequence:
                        continue
                    temp_score = 1
                    for position, base in enumerate(subsequence):
                        temp_score = temp_score*PFM_dictionary[motif][position,base2index[base]]/max(PFM_dictionary[motif][position,:])
                    if np.round(temp_score,2) > score_cutoff[motif]:
                        f.write('%s\t%s\t%s\t%.4f\n' %(seq_name.replace('-','_'), motif, i+1, temp_score))
    output = pd.read_csv('element_output.txt',sep='\t',header=None)
    output.columns = ['sequence','motif','position','score']
    output[['score']] = output[['score']].astype(float)
    
    return output

def ICdictionary(PWM):
    """Create dictionary that maps TF to the information content of its PWM

    Arguments
    ----------
    PWM : dictionary
        A dictionary that maps TF to its position weight matrix (PWM).

    Returns
    -------
    IC : dictionary
        A dictionary that maps a TF to the information content of its PWM.

    Notes
    -----
    Use readFASTA to create a dictionary of PWMs first.
    ICdictionary assumes that a pseudocount has already been added.

    Examples
    --------
    >>> PWM,checkFBgn,PWMnames = readFASTA("Datasets/UmassPGFE_PWMfreq_PublicDatasetA_20150901.txt", \
                          'PWM','v',re.compile('(FBgn\d*)'))
    >>> IC = ICdictionary(PWM)
    >>> IC['cad']
    6.455107925364187
    """
    IC = {} # Information Content dictionary
    tempIC = [] # temporary list of the info content at each position in PWM
    q = [0.297, 0.203, 0.203, 0.297] # background frequencies for DNA bases in intergenic region
                                     # The order of the bases is: A C G T

    # Calculating Information Content of PWMs
    for key in PWM:
        for line in PWM[key]:
            # Note: pseudocounts already added
            # calculate p = frequencies of each base at this position in motif
            p = [float(x)/sum(line) for x in line]
            pq = zip(p,q)
            tempIC.append(sum([x[0]*math.log(x[0]/x[1])/math.log(2) for x in pq]))
        IC[key] = sum(tempIC)
        tempIC = []

    return IC

##########################################################
# PATSER-RELATED PROCESSING Functions
##########################################################

def createSeqIN(dictionary,filter_string,regex,filepath,filename,write,FastA):
    """Create input sequence file for patser and calculate the length of sequences

    Arguments
    ----------
    dictionary : dictionary
        dictionary that maps an identifier to a (DNA) sequence.
    filter_string : string
        string that must be in the sequence name to be included in the patser input
        sequence file
        0 if none (i.e. all sequences in dictionary will be put in patser input sequence
        file)
    regex : compiled regular expression, i.e. re.compile(pattern)
        used to parse the sequence name for the desired identifier for each key
        in the dictionary (e.g. for the AP/DV enhancers from ap_enc_12.fa and dv_enc_12.fa,
        regex = re.compile('([\w\-]+)_mel'))
        0 if unneeded (i.e. all sequence names are fine)
    filepath : string
        path to file in which the patser input sequences are saved
    write : integer
        1 to write to file (in patser-input format)
        0 otherwise
    FastA : integer
        1 to write to file in FastA format
        0 otherwise

    Returns
    -------
    od_len_sequence : dictionary
        A dictionary that maps the sequence name to its length

    Notes
    -----
    First step in the process of running set of sequences through patser and processing
    the patser output

    Examples
    --------
    >>> od_len_ap_enhancer = createSeqIN(AP_enhancers,'mel',re.compile('([\w\-]+)_mel'),'patserIn/AP_enhancers_mel',1,0)
    >>> od_len_ap_enhancer['eve_2']
    880
    >>> minVT_enhancer_len = createSeqIN(minVT_enhancers_ALL,0,0,'patserIn/MinVT_enhancers_mel',0,0)
    >>> minVT_enhancer_len['VT26168.1']
    207
    """
    # create patserIn directory if it does not exist
    if not os.path.exists('%s/patserIn' %filepath):
        os.makedirs('%s/patserIn' %filepath)
    # filter out unwanted sequences from input dictionary (if needed)
    if filter_string != 0:
        filtered_dict = {key:value for (key,value) in dictionary.items() \
                            if filter_string in key}
    else:
        filtered_dict = dictionary
    # order keys in filtered dictionary
    filtered_dict = collections.OrderedDict(sorted(filtered_dict.items(),key=lambda t:t[0]))

    lenSequence = {} # dictionary of length of sequences
    if write == 1:
        with open('%s/patserIn/%s.txt' %(filepath,filename),'w+') as f:
            for key,value in filtered_dict.items():
                if regex is not 0:
                    m = regex.search(key)
                    if m:
                        key = m.group(1).replace('-','_')
                    else:
                        print("%s doesn't have expected regex." %key)
                if FastA is 1:
                    temp = '>%s\n%s\n' %(key,value)
                else: # patser input format
                    temp = '%s \ %s \ \n' %(key,value)
                lenSequence[key] = len(value)
                f.write(temp)
    else:
        for key,value in filtered_dict.items():
            lenSequence[key] = len(value)
    od_len_sequence = collections.OrderedDict(sorted(lenSequence.items(),key=lambda t:t[0]))
    return od_len_sequence


def runPatser(filepath,origFile,TF_list,complementary=1,print_restrict='0',percentile=75,aligned=0):
    """Run patser on the input sequences (generated by createSeqIN)

    Arguments
    ----------
    filepath : string
        path to the directory that contains all the results
    origFile : string
        name of input sequences file; runPatser expects these input sequence files to be
        in patserIn directory
    TF_list : list
        list of TFs who position weight matrices should be used to score all subsequences
        of length k (k = motif length)
    complementary : int
        1 (default) for scoring the complementary strand
        0 do not score the complementary strand
    print_restrict : string, int, dict
        restrictions on which scores are printed in the patser output file
        if type is string, '0' (default) to print scores > 0
                            '1' to print all scores
        if type is int, print this number of top scores
        if type is dict, which maps TF to the max ln(p-value), print scores with
        ln(p-values) less than the given value for each TF
    percentile : int
        if dict of ln(p-values) given (for print_restrict), then this integer is the
        percentile of the ln(p-values) of the aligned sequences used to create the PWMs
        that is being used a the threshold for real vs false motif hits.
    aligned : string, int
        if type is string, the type of input sequence is special, e.g. 'Aligned' or 'Raw'
            the input sequence file will be in the format: 'patserIn/%s_%s.txt' %(origFile,tf)
            because all aligned or raw sequences would be associated with a single TF
        if type is int, 0 (default) indicates "normal" input sequence
            the input sequence file will be in the format: 'patserIn/%s.txt' %origFile

    Output
    -------
    patser output files : text file
        For each TF, Patser will generate an output file that contains the position, score, and
        ln(p-value) of each subsequence that satisfies the print restrictions that you
        have given

    Notes
    -----
    Patser scores k-mers (subsequences of length k) against the given position weight
    matrix

    Examples
    --------
    >>> runPatser('MinVT_enhancers_mel',TF_stage_ALL,print_restrict='0',percentile=75,aligned=0)
    """
    path_to_patser = '/Applications/patser-v3e.1/patser-v3e'
    # lnp = str(np.log(0.001)) # reasonable choice for threshold if fixed
    # Note: pseudocount already added to PWM; no need to add again --> "-b 0"
    if not os.path.exists('%s/patserOut' %filepath):
        os.mkdir('%s/patserOut' %filepath)

    for tf in TF_list:
        matrix = '%s/matrix/%s.txt' %(filepath,tf) # location of TF PWM file
        if type(aligned) is str:
            inputFile = '%s/patserIn/%s_%s.txt' %(filepath,origFile,tf) # input sequences
        else:
            inputFile = '%s/patserIn/%s.txt' %(filepath,origFile) # input sequences
#        if type(print_restrict) is dict and len(np.unique(print_restrict.values())) > 1:
#            outputFile = 'patserOut/%s_%s_aligned%s.txt' %(origFile,tf,percentile) # PATSER output
#        else:
#            outputFile = 'patserOut/%s_%s.txt' %(origFile,tf) # PATSER output
        if type(print_restrict) is dict: # ln(p-values)
            if len(np.unique(print_restrict.values())) == 1: # if using fixed ln(p-value)
                outputFile = '%s/patserOut/%s_%s_lnp%s.txt' %(filepath,origFile,tf,print_restrict.values()[0]) # PATSER output
            else:
                outputFile = '%s/patserOut/%s_%s_aligned%s.txt' %(filepath,origFile,tf,percentile) # PATSER output
            cmd = [path_to_patser, '-A', 'a:t', '0.297', 'c:g', '0.203', \
                   '-p', '-m', matrix, '-b', '0', '-c','-lp', str(print_restrict[tf]), \
                   '-d1', '-f', inputFile]
        elif type(print_restrict) is int: # top scores
            outputFile = '%s/patserOut/%s_%s_topscores%s.txt' %(filepath,origFile,tf,print_restrict) # PATSER output
            cmd = [path_to_patser, '-A', 'a:t', '0.297', 'c:g', '0.203', \
                   '-p', '-m', matrix, '-b', '0', '-c','-t',str(print_restrict), \
                   '-d1', '-f', inputFile]
        elif print_restrict == '0': # scores > 0
            outputFile = '%s/patserOut/%s_%s.txt' %(filepath,origFile,tf) # PATSER output
#             outputFile = 'patserOut/%s_%s_ls0.txt' %(origFile,tf) # PATSER output
            cmd = [path_to_patser, '-A', 'a:t', '0.297', 'c:g', '0.203', '-ls','0', \
                   '-p', '-m', matrix, '-b', '0', '-c', '-d1', '-f', inputFile]
        else: # print all scores
            outputFile = '%s/patserOut/%s_%s_all.txt' %(filepath,origFile,tf) # PATSER output
            cmd = [path_to_patser, '-A', 'a:t', '0.297', 'c:g', '0.203', \
                   '-p', '-m', matrix, '-b', '0', '-c', '-d1', '-f', inputFile]

        # if ' ' in filepath:
        #     outputFile = outputFile.replace('\ ',' ')
        if complementary == 0:
            cmd.pop(13)
            # print(cmd)

        process = subprocess.call(cmd,stdout = open(outputFile, 'w'))
        print('Done: %s' %tf)

def parsePatser(filepath,TF_list,patserFile,lnp,category,percentile=75,input_extension='.txt'):
    """Reads in patser output and saves putative binding sites that satisfy the ln(p-value)
    cutoff in a dataframe

    Arguments
    ----------
    filepath : string
        path to the directory that contains all the results
    TF_list : list
        list of TFs of interest
    patserFile : string
        name of the input sequence file; parsePatser will read in patser output files, e.g.
        'patserOut/AP_enhancers_mel_bcd.txt' where patserFile = 'AP_enhancers_mel'
    lnp : dictionary
        dictionary that maps TF to its maximum ln(p-value)
    category :
        name of set of enhancers; parsePatser will save compiled patser output in file
        'patserAnalysis/%s_enhancers_patser_output%s'%(category,extension)
    percentile : integer
        if dict of ln(p-values) given (for print_restrict), then this integer is the
        percentile of the ln(p-values) of the aligned sequences used to create the PWMs
        that is being used a the threshold for real vs false motif hits.
    input_extension : string
        the last portion of the filename of the patser output file (depends on what
        restrictions were placed on the scores that were printed), so a patser output file
        of 'patserOut/AP_enhancers_ab_aligned75.txt' would have an input_extension of
        '_aligned75.txt'

    Output
    -------
    compiled patser output files : text file
        A tab-delimited file containing info about putative binding sites predicted by
        patser for the TFs of interest that satisfy the ln(p-value) cutoff given,
        including enhancer, tf, position at which it is located, score, and the ln(p-value)
    output : dataframe
        A data frame containing the data from the tab-delimited file

    Notes
    -----

    Examples
    --------
    >>> minVT_patser_output = parsePatser(TF_stage_ALL,'MinVT_enhancers_mel',lnp,'MinVT')
    """
    if not os.path.exists('%s/patserAnalysis' %filepath):
        os.mkdir('%s/patserAnalysis' %filepath)

    TFs_skipped = []
    output = pd.DataFrame(columns=['sequence','motif','position','complement','score','lnp'])
    # ctr = 0 # row number of dataframe
    # regular expression to parse a line of patser output
    regex = re.compile('([\w\-\.\(\)]+)\s+position=\s+([\dC]+)\s+score=\s+([\d\.]+)\s+ln\(p-value\)=\s+([\-\.\w]+)')
    # group(1) = enhancer, or sequence name
    # group(2) = position
    # group(3) = score
    # group(4) = ln(p-value)
    if len(np.unique(lnp.values())) == 1:
        extension = '.txt'
    else:
        extension = '_aligned%s.txt' %percentile
    with open('%s/patserAnalysis/%s_patser_output%s'%(filepath,category,extension),'w') as f:
        for tf in TF_list:
            with open('%s/patserOut/%s_%s%s' %(filepath,patserFile,tf,input_extension),'r') as f1:
                for line in f1:
                    if regex.search(line):
                        m = regex.search(line)
                        # try:
                        if float(m.group(4)) < lnp[tf]:
                            if 'C' in m.group(2):
                                f.write(m.group(1).replace('-','_') + '\t' + tf + '\t' + \
                                    m.group(2)[:-1] + '\t' + '1' + '\t' + m.group(3) + '\t' + m.group(4) + '\n')
                            else:
                                f.write(m.group(1).replace('-','_') + '\t' + tf + '\t' + \
                                    m.group(2) + '\t' + '0' + '\t' + m.group(3) + '\t' + m.group(4) + '\n')                                
                        # except KeyError:
                        #     if tf not in TFs_skipped:
                        #         TFs_skipped.append(tf)
            print(tf)
    # print out the TFs that have been skipped (because their PWM does not have aligned sequences)
    if len(TFs_skipped) > 0:
        print('No aligned sequences available for %s.' %TFs_skipped)
    output = pd.read_csv('%s/patserAnalysis/%s_patser_output%s' %(filepath,category,extension),sep='\t')
    output.columns = ['sequence','motif','position','complement','score','lnp']
    output[['position','complement']] = output[['position','complement']].astype(int)
    output[['score','lnp']] = output[['score','lnp']].astype(float)
    # write patser output to file
#   output.to_csv('patserAnalysis/%s_patser_output%s'%(category,extension),sep='\t',index=False)
    return output

### ADD STEP 3.5!!!
def noDuplicates(output,category,PWM_offset,filepath):
    """Concatenates references to the same binding site by removing repeated references
    and adding the TF to a single reference to that binding site

    Arguments
    ----------
    output : dataframe
        dataframe containing info about putative binding sites predicted by patser for the
        TFs of interest that satisfy the ln(p-value) cutoff given, including enhancer, tf,
        position at which it is located, score, and the ln(p-value)
        Note: output of parsePatser
    category : string
        name of set of enhancers; parsePatser will save compiled patser output in file
        'patserAnalysis/%s_enhancers_patser_output%s'%(category,extension)
    PWM_offset : dictionary
        dictionary mapping TF to its heterodimeric partners and the number of base pairs
        by which the partners' motif is shifted compared to the TF

    Output
    -------
    revised patser output files : text file
        A tab-delimited file containing info about putative binding sites predicted by
        patser for the TFs of interest that satisfy the ln(p-value) cutoff given,
        including enhancer, tf, position at which it is located, score, and the ln(p-value)
        Here, all duplicate references to binding sites have been removed, and the single
        reference to a binding site that may bind multiple heterodimers includes all
        possible heterodimeric TFs in the TF column
    output : dataframe
        A data frame containing the data from the tab-delimited file

    Notes
    -----

    Examples
    --------
    >>> minVT_output_noDuplicates = noDuplicates(minVT_output,'MinVT',PWM_offset)
    """

    for enhancer in output.enhancer.unique():
        print(enhancer)
        for tf in PWM_offset.keys():
            temp_tf = output[(output.enhancer == enhancer) & (output.tf == tf)]
    #         print tf
            for pos,idx in zip(temp_tf.position,temp_tf.index.tolist()):
                for partner,offset in PWM_offset[tf].items():
                    if 'C' in pos:
                        temp = output[(output.enhancer == enhancer) \
                                    & (output.tf == partner) \
                                    & (output.position == str(int(pos.replace('C','')) + offset[1])+'C')]
                    else:
                        temp = output[(output.enhancer == enhancer) \
                                    & (output.tf == partner) \
                                    & (output.position == str(int(pos) + offset[0]))]
                    if not temp.empty:
                        print('%s,%s repeat!' %(tf,partner))
                        output.ix[idx,'tf'] += ',%s' %partner
                        output = output.drop(temp.index.tolist()[0])
                        output.to_csv('%s/patserAnalysis/%s_enhancers_patser_output_noduplicates.txt' %(filepath,category))
    output.to_csv('%s/patserAnalysis/%s_enhancers_patser_output_noduplicates.txt' %(filepath,category))
    return output

# STEP FOUR: pull out the average IC of the enhancers, number of TFs in enhancers -- allow for counting of duplicates when only one is present
def PatserAnalysis(patser_output,enhancers,TF_list,heterodimers,IC):
    """ takes dataframe of patser output and calculates the number of TF binding sites and
    the average information content of each enhancer

    Arguments
    ----------
    patser_output : dataframe
        dataframe that contains the enhancer, tf, position, score and ln(p-value) of all
        putative transcription factor binding sites that satisfied the ln(p-value) cutoff
        given
    enhancers : list or string
        'all' if all enhancers in patser output dataframe should be included in analysis
        list of enhancers of interest if only a subset of enhancers should be included in
        analysis
    TF_list : list
        list of TFs of interest
    heterodimers : dictionary
        dictionary mapping TFs to heterodimeric partners
    IC : dictionary
        dictionary mapping TF to the information content of the TF's PWM/motif

    Output
    -------
    od_av_enhancer_IC : dictionary
        A dictionary mapping enhancers to the average information content of that enhancer
    od_enhancer_nTFs : dictionary
        A dictionary mapping enhancers to the total number of TFs in that enhancer
    TF_counts : dataframe
        A dataframe detailing enhancers and the numbers of binding sites of each TF
        predicted

    Notes
    -----

    Examples
    --------
    >>> temp_av_IC, temp_nTFs, temp_TF_counts \
     = PatserAnalysis(minVT_patser_output,minVT_enhancers,TF_stage,heterodimers,IC)

    """

    # add TFs that bind to the same site to TF list
    for tfs in patser_output.tf:
        if ',' in tfs:
            if set(tfs.split(',')+[heterodimers[tf] for tf in tfs.split(',')]).issubset(TF_list) and tfs not in TF_list:
                TF_list.append(tfs)
    # remove TF from TF list if heterodimeric partner is not present
    TF_list = list(set(TF_list)-set([TF for TF in TF_list if TF in heterodimers if heterodimers[TF] not in TF_list]))
    TF_counts = patser_output[patser_output['tf'].isin(TF_list)].groupby(['enhancer','tf']).count()['position'].to_frame()
    TF_counts = TF_counts.reset_index(level=['enhancer', 'tf'])
    TF_counts.rename(columns = {'position':'counts'}, inplace = True)

    if enhancers == 'all':
        enhancers = TF_counts.enhancer.unique()

    enhancer_IC = {}
    enhancer_nTFs = {}
    av_enhancer_IC = {}
    for enhancer in enhancers:
        temp = TF_counts[TF_counts.enhancer == enhancer]
        if temp.empty: #if no predicted binding sites in this enhancer
            print('%s has no predicted binding sites' %enhancer)
        else:
            for tf in temp.tf.unique():
                # weighted total info content (if multiple heterodimeric TFs have the same binding site, then average info content)
                if enhancer in enhancer_IC:
                    enhancer_IC[enhancer] += float(temp.counts[temp.tf == tf])*np.mean([np.power(2,-IC[x]) for x in tf.split(',')])
                else:
                    enhancer_IC[enhancer] = float(temp.counts[temp.tf == tf])*np.mean([np.power(2,-IC[x]) for x in tf.split(',')])
            enhancer_nTFs[enhancer] = sum(temp.counts)
            av_enhancer_IC[enhancer] = enhancer_IC[enhancer]/enhancer_nTFs[enhancer]
    # Alphabetize IC and overall TF counts dictionary
    od_av_enhancer_IC = collections.OrderedDict(sorted(av_enhancer_IC.items(),key=lambda t:t[0]))
    od_enhancer_nTFs = collections.OrderedDict(sorted(enhancer_nTFs.items(),key=lambda t:t[0]))
    return od_av_enhancer_IC, od_enhancer_nTFs, TF_counts

# STEP FIVE: export dictionary data
# INPUT:
#       dictionary = name of dictionary
#       filename = save to this file (as string)
#       multiline = 0 (single line), 1 (multiple lines)
def dictionaryExport(dictionary,filename,multiline):
    with open(filename,'w') as f:
        for k,v in dictionary.items():
            if multiline == 0:
                f.write(str(k)+'\t'+'%.10f'%v+'\n')
            elif multiline == 1:
                f.write('>'+str(k)+'\n'+str(v)+'\n')

# STEP SIX: import dictionary data
# INPUT:
#       filename = name of file to be imported (as string)
#       ordered = 1 if the dictionary is ordered
# OUTPUT:
#       dictionary
def dictionaryImport(filename,ordered):
    dictionary = {}
    regex = re.compile('([\w\.\-]+)\t([\w.]+)')
    with open(filename,'r') as f:
        for line in f:
            if regex.search(line):
                m = regex.search(line)
                dictionary[m.group(1)] = float(m.group(2))
    if ordered is 1:
        dictionary = collections.OrderedDict(sorted(dictionary.items(),key=lambda t:t[0]))
    return dictionary    
