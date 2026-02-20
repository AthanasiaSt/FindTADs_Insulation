#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:02:22 2020

@author: zelda
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


###----store the data into a dataframe----####

data = pd.read_csv('interactions_HindIII_fdr0.01_intra.txt', sep=" ", header=None, engine='python', delimiter = "\t")

df = data.iloc[1:] #drop the first row of the dataframe

df = df.apply(pd.to_numeric) # make all the values numeric


######--------making the symmetric contact fequency matrix--------######
def squareContact_matrix(df, chromosome):
    
    #initiation of lists
    loci_final = [] #store the loci that will be placed on the horizontal and vertical axis of the matrix 
    loci_chr = [] #store the corresponding chromosome for each locus that will be placed on the horizontal and vertical axis of the matrix 
    
    for i in range(1,chromosome+1): #take the data for each chromosome separately 
    
        sorted_loc2 = (df.loc[df[2] == i, 3]).to_numpy() #making the dataframe columns, that store the loci, into arrays
        sorted_loc1 = (df.loc[df[0] == i, 1]).to_numpy()
        
    
    #####------find the unique elements of each loci for each chromosome separately-----######     
        loci_uniq = list(np.unique(sorted_loc1)) + list(np.unique(sorted_loc2))#store the unique loci1 and loci2
                                                                               #into a temporary list
        loci_uniq = np.sort(np.unique(np.array(loci_uniq)))#sort the list with the unique loci1 and loci2       
        
        loci_final += list(loci_uniq) #store the loci in a final list (loci_final)
            
        loci_chr += [i]*(len(loci_uniq)) #make a list with the chromosome index for the corresponding loci
     
        
    ####-----Making the symmetric matrix-------########
    
    N = len(loci_final) # the first column of the matrix (containing all unique loci)

    matrix = np.zeros((N+2, N+2), dtype=int) #initiation of the matrix, zeros placed in the positions of no Frequency data
    
    matrix[2:,0] = loci_chr # the corresponding chromosome index in the first column
    
    matrix[2:,1] = loci_final #the loci in the second column 
    
    matrix[0,2:] = loci_chr #the corresponding chromosome index in the first row
    
    matrix[1,2:] = loci_final#the loci in the second row 
    
    matrix.shape
    for chrm in range(1,chromosome+1):
        
        pair = df.loc[df[0] == chrm, [1,3,4]] #for each chromosome take the pair of loci1,loci2 and frequency columns of the dataframe
        
        for loci in matrix[2:,1]: 
            
            loci_freq = pair.loc[pair[1] == loci, [3,4]] #for each loci1 find the corresponding loci2 and frequency in the dataframe
            
            indices = pair.index[pair[1] == loci].tolist()#find the corresponding indices in the dataframe
            
            index_loc1 = np.where(matrix[:,1] == loci)[0][0]#find the loci in the matrix 
            
            for index in indices: #use the indices to take the corresponding elements of the dataframe
                #and store them in the corresponding matrix positions
                index_loc2 = np.where((matrix[0,:] == chrm) & (matrix [1,:] == loci_freq[3][index]))[0][0]
                
                matrix[index_loc1,index_loc2] = loci_freq[4][index] #fill the symmetric positions of the matrix
                matrix[index_loc2, index_loc1] = loci_freq[4][index]


    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111)
    ax.set_title('Contact Matrix')
    plot=plt.spy(matrix[2:,2:], markersize=0.005,precision=0,color='red',label='Contact Frequency') #casual plotting
    ax.set_xlabel('Chromosome loci index')
    ax.set_ylabel('Chromosome loci index')
    ax.legend(loc='best')
    plt.show()
    
    # fig = plot.get_figure()
    # fig.savefig('Square_contact_matrix.png',dpi=100)
    
    return (matrix)

matrix=squareContact_matrix(df, 16)


from scipy.signal import savgol_filter
from scipy import stats
import sys
    

def TAD_boundaries(matrix, chromosome):

    #making a "SUB"matrix for each chromosome separately
    
    N = np.where(matrix[0,:] == chromosome)[0] #Indices of the loci of the corresponding chromosome in the matrix 
    
    M = np.zeros([N[-1]+2-N[0],N[-1]+2-N[0]]) #initiate a matrix with the right dimensions 
    
    M[0,1:] = matrix[1,N[0]:N[-1]+1] #in the first row --> loci
    M[1:,0] = matrix[N[0]:N[-1]+1,1] #in the first column --> loci
    
    M[1:,1:]= matrix[N[0]:N[-1]+1,N[0]:N[-1]+1] #in the remaining positions of the matrix --> corresponing frequencies
    
    ####----Plotting the "SUB"matrix----####
    ####---heatmap-----####
    sns.set()
    ax = sns.heatmap(M[1:,1:], vmin=0, vmax=np.max(M[1:,1:]))
    ax.set(title=('Frequency square contact matrix, Chromosome:{chromosome}'.format(chromosome=chromosome)),xlabel="index of loci",ylabel="index of loci")
    # fig = ax.get_figure()
    # fig.savefig('Chromosome{chromosome}_contact.png'.format(chromosome=chromosome),dpi=300)
        
    #####-----sparse plot-------#####
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111)
    ax.set(title=('Frequency square contact matrix, Chromosome:{chromosome}'.format(chromosome=chromosome)),xlabel="index of loci",ylabel="index of loci")
    plot=plt.spy(M[1:,1:], markersize=0.8,precision=0,color='red',label='Contact Frequency') 
    ax.legend(loc='best')
    plt.show()
    
    # fig = plot.get_figure()
    # fig.savefig('Chromosome{chromosome}_contact_2.png'.format(chromosome=chromosome),dpi=300)
        
    
    #####----Identify TAD boundaries-----####
    
    #For each bin (locus) generate an insulation score, the bins with not many frequency-contacts around them are potential TAD boundaries
    
    window = int(len(N)/8)-1 #take  a range of loci upstream and downstream of the bin
    averagebin_freq=[] #to store the insulation scores 
    bin_index = [] #to store the index of the bins 
    
    N = abs((M.shape[1]-window+1) - (1+window)) #dimensions for the average matrix
    D = window+1 # in which each vector with the sum of each row of the matrix bin_ will be kept 
    average_matrix = np.zeros([N,D])

    
    for i in range(1+window, M.shape[1]-window): #take each locus as bin (start + window and end - window)
        
        up = i - window #upstream of the bin
        down = i + window +1 #downstream of the bin
        
        bin_ =  M[up:i+1,i:down] #making a "square" with its lower left angle touching the main diagonal
          
        averagebin_freq +=[(np.sum(np.sum(bin_, axis=1)))/(bin_.shape[0])] #add the frequencies in the rows and divide with the number of loci taken
        
        bin_index += [M[i,0]] #store the locus being the bin in each loop 
        
        average_matrix[i-window-1,:] = np.sum(bin_, axis=1).reshape(1,-1)

    
    ###---normalization of the insulation scores----#####
    ###----log2(insulation scores)/mean(all insulation scores)-----#####
    
    averagebin_freq=np.array(averagebin_freq) 
    index = np.where(averagebin_freq==0.0) #if there is a zero average frequency make it into ONE(1) 
    averagebin_freq[index]=1
    
    norm_insu_scores = np.log2(averagebin_freq)/np.mean(averagebin_freq) #log2 of each insulation score 
                                                                            #and division by the mean of all insulation scores
######-----plotting-----####                                                  #deriving from each chromosome
    sns.set()    
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(12,7))                                    
    ax = sns.lineplot(bin_index, norm_insu_scores)
    ax.set(title=('Normalized insulation scores across the bins of Chromosome:{chromosome}'.format(chromosome=chromosome)),xlabel="position of bin",ylabel="Normalized insulations score")
    # fig = ax.get_figure()
    # fig.savefig('Chromosome{chromosome}_insulation.png'.format(chromosome=chromosome),dpi=300)
        
    
    ######----smoothing the line of insulation scores-------#####
    
    x = np.array(bin_index)
    y = np.array(norm_insu_scores)
    
    if window%2 == 0: #defining the parameter for smoothing, has to be an odd number, close to the window
        parameter = window+1
        
    else:
        parameter=window
        
    Smoo_norm_insu_scores = savgol_filter(y, parameter, 3) # polynomial order 3
     
 #####plotting######
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.legend(loc='best')
    ax.plot(x, y, label='Insulations scores')
    ax.plot(x, Smoo_norm_insu_scores, color='red', label='Smoothed insulation scores')
    legend = ax.legend(loc='best', shadow=True, fontsize='medium')
    
    ax.set(title=('Insulation scores, Chromosome:{chromosome}'.format(chromosome=chromosome)),xlabel="position of bin",ylabel="Insulation scores")
    # fig = ax.get_figure()
    # fig.savefig('Chromosome{chromosome}_insulation.png'.format(chromosome=chromosome),dpi=300)
            
    ######------Identifying the minima of the insulation score curve-----######
    
    ###----Delta vector-----#####
    #Delta vector is made by subtracting the insulation score of the bin upstream from that of the reference bin
    #and then adding that to the substraction of the insulation score of the bin downstream to the reference bin.....
    #------!!!!using only one bin to the left and one bin to the right of the reference bin !!!!!----#####
    
    delta_raw=[] #to store the delta vector
    delta_index_2=[] # to store the corresponding indices of the reference bin  
    
    for i in range(1,len(Smoo_norm_insu_scores)-1):
        
        delta_raw += [(Smoo_norm_insu_scores[i-1]-Smoo_norm_insu_scores[i]) - (Smoo_norm_insu_scores[i+1]-Smoo_norm_insu_scores[i])]
        delta_index_2+=[bin_index[i]] #keep the reference bin index
   
   ######----smoothing the line of Delta vector-------#####
    if window%2 == 0:
        parameter = window+1
    else:
        parameter=window
        
    delta = savgol_filter(delta_raw, parameter, 3) #polynomial order 3
      
#####----plotting-----#####
    sns.set() 
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(12,7))  
    ax = sns.lineplot(delta_index_2, delta_raw, label='Delta raw')
    ax.axhline(0, ls='--',color='r')  
    ax.plot(delta_index_2, delta, color='red', label='Smoothed Delta vector')
    legend = ax.legend(loc='best', shadow=True, fontsize='medium') 
    ax.set(title=('Delta vector across the reference bins of Chromosome:{chromosome}'.format(chromosome=chromosome)),xlabel="index of bin",ylabel="score of delta vector")
    # fig = ax.get_figure()
    # fig.savefig('Chromosome{chromosome}_DElta.png'.format(chromosome=chromosome),dpi=300)
            
    
    ###---find the bins which are before and after the zero crossings of delta vector-----#####
    #####---keep the bins that are closest to zero----#####
    
    delta=np.array(delta)
    
    zero_crossings_before = np.where(np.diff(np.signbit(delta)))[0] #find the indices of the elements of delta before the point
                                                             #at which there is a sign change                                
    zero_crossings_after = zero_crossings_before + 1 #the values after the sign change
    
    zero_crossings=[]
    for i in range(len(zero_crossings_after)): #choose the values next to zero crossings that are closest to zero 
        if abs(zero_crossings_after[i]-0) < abs(zero_crossings_before[i]-0):
            zero_crossings += [zero_crossings_after[i]]
        else:
            zero_crossings += [zero_crossings_before[i]]


    minima_crossings=[]
    maxima_crossings=[]
    for i in range(len(zero_crossings)): #keeping only the points of delta on a descending line, next to the zero crossing
                                          # and take the corresponding bins---checking if the next delta value is lower 
                                          # or higher
        elem = zero_crossings[i]
        elem_after = zero_crossings[i] + 1
        
        if delta[elem] > delta[elem_after]:
            minima_crossings+=[elem]        
        else:
            maxima_crossings+=[elem]    

    
    delta_index_2=np.array(delta_index_2)
    minima = delta_index_2[minima_crossings] #potential boundaries of TADS
    maxima=delta_index_2[maxima_crossings] #potential TAD regions
 
    #### ------ evaluation of minima ---- #####
    
    ###Boundary strength------cannot be computed for the last minimum-----#####
    ## The boundary strength was defined as the difference in the delta vector
     #between the local maximum to the left and local minimum to the right of the boundary bin

    print('Information on Chromosome:',chromosome)

    boundary_strength=np.zeros([len(minima)]) #to store the boundary strengths
    for i in range (len(minima)-1):
        
        index = np.where(zero_crossings == minima_crossings[i])[0][0]
        print('\nFor the current boundary bin:',minima[i])
    
        maximum_left = delta[zero_crossings[index-1]]
        print('The local maximum to the left is:',delta_index_2[zero_crossings[index-1]] )
    
        minimum_right = delta[minima_crossings[i+1]]
        print('The local minimum to the right is:',delta_index_2[minima_crossings[i+1]] )
    
        print('\nThe boundary strength is:')
        
        print('---------',abs(maximum_left-minimum_right),'-----------\n')
 
        boundary_strength[i]=(abs(maximum_left-minimum_right))
        
    ###---plotting the boundary strengths-----####
    sns.set()      
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(12,7))                                   
    ax = sns.barplot(minima, boundary_strength)
    ax.set(title=('Boundary strength, chromosome:{chromosome}'.format(chromosome=chromosome)),xlabel="bin",ylabel="score of Boundray strength")
    # fig = ax.get_figure()
    # fig.savefig('Chromosome{chromosome}_boundary.png'.format(chromosome=chromosome),dpi=300)

    print('\n!!!!!For the last minimum, the boundary strength cannot be computed!!!!')
    
    #####-----t-test--------#######
    ##check if the average_frequency (not-normalized) score for the minima are significantly different 
    # from that of the maximum upstream and the maximum downstream of the reference bin
    

    N=len(minima)
    D=6 
    Final_matrix = np.zeros([N,D]) #store the final information in a matrix
    # Chromosome, minimum position, pvalues, boundary strength
    print("-------Pvalues-------")

    for i in range (len(minima)):
        maximum_up=0
        maximum_down=0
        
        index_minima = np.where(x == minima[i])[0][0] 
        vector_minima = average_matrix[index_minima,:] #the vector of the minimum

        index = np.where(zero_crossings == minima_crossings[i])[0][0]
        print('\nFor the local minimum',minima[i])
        
        try: #checking if there is a maximum upstream and downstream of the minimum
            if index-1 < 0:
                maximum_up = 0
            else:
                print('\nThe upstream local maximum is:',delta_index_2[zero_crossings[index-1]])
                maximum_up = delta_index_2[zero_crossings[index-1]]
                
                maximum_down = maxima[(np.where(maxima == maximum_up)[0][0])+1]
                print('The downstream local maximum is:',maximum_down)
           
            if maximum_up == 0:
                maximum_down = maxima[0]
                print('The downstream local maximum is:',maximum_down)
                index_maximum_down = np.where(x == maximum_down)[0][0]
                vector_maximum_down =  average_matrix[index_maximum_down,:]
                print('There is no upstream local maximum')
                 
                Final_matrix[i,0] = chromosome  
                Final_matrix[i,1] = minima[i]  
                
                pvalue_down = stats.ttest_ind(vector_minima,vector_maximum_down)[1]    
                print('The pvalue resulting from the comparison with the  local maximum downstream of the minimum is:',pvalue_down)
                Final_matrix[i,3] = pvalue_down
                continue
            
            elif maximum_down == 0:
                print('THere is no downstream maximum')
                index_maximum_up = np.where(x == maximum_up)[0][0]
                vector_maximum_up =  average_matrix[index_maximum_up,:]
                
            else:
                     
                maximum_down = maxima[(np.where(maxima == maximum_up)[0][0])+1]
            
                index_maximum_up = np.where(x == maximum_up)[0][0]
                index_maximum_down = np.where(x == maximum_down)[0][0]
            
                vector_maximum_up =  average_matrix[index_maximum_up,:]
                vector_maximum_down =  average_matrix[index_maximum_down,:]
            
            
        except IndexError:
            print("Oops!", sys.exc_info()[0], "occurred.")
            print('Only one local maximum is available for comparison')
            
            Final_matrix[i,0] = chromosome  
            Final_matrix[i,1] = minima[i]  
     
            if maximum_down ==0:
                pvalue_up = stats.ttest_ind(vector_minima,vector_maximum_up)[1] 
                print('\nThe pvalue resulting from the comparison with the local maximum upstream of the minimum is:',pvalue_up)
                Final_matrix[i,2] = pvalue_up

            Final_matrix[i,5] = boundary_strength[i]

            continue

        
        pvalue_up = stats.ttest_ind(vector_minima,vector_maximum_up)[1] 
        print('\nThe pvalue resulting from the comparison with the local maximum upstream of the minimum is:',pvalue_up)
        
        pvalue_down = stats.ttest_ind(vector_minima,vector_maximum_down)[1]    
        print('\nThe pvalue resulting from the comparison with the local maximum downstream of the minimum is:',pvalue_down)

        pvalue_maxima = stats.ttest_ind(vector_maximum_up,vector_maximum_down)[1]    
        print('\nThe pvalue resulting from the comparison of the two local maxima:',pvalue_maxima,'\n')

        Final_matrix[i,0] = chromosome #Final matrix to store the resulting information for each 
        Final_matrix[i,1] = minima[i]  # local minimum that may be a TAD boundary
        Final_matrix[i,2] = pvalue_up
        Final_matrix[i,3] = pvalue_down
        Final_matrix[i,4] = pvalue_maxima
        Final_matrix[i,5] = boundary_strength[i]
 
    results_df = pd.DataFrame(Final_matrix)
    results_df.columns=['Chromosome','bin', "pvalue_left", "pvalue_right","pvalue_maxima", "boundary_strength"]

    return results_df  

#results_df = TAD_boundaries(matrix,16)

###----store the Dataframes with the resulting information on each chromosome in a list of 16 dataframes------#####
lst_dfs=[]
for i in range(1,17):
    print("--------------------Chromosome:",i)
    results_df = TAD_boundaries(matrix,i)
    lst_dfs += [results_df]

#return a csv file that contains all dataframes    
with open (r'TAD_Boundaries_results.csv', 'a') as f:
    for df in lst_dfs:
        
        df.to_csv(f, sep='\t')
        
