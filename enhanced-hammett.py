#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats


def calc_m(data):
    """ Create the A matrix for the overdetermined system of equations Ax=0
            in this matrix, each entry is weighted by the variance of m
            The output matrix has a column for each reaction XY and a row for each pair of reaction XY_X'Y'
            If XY == X'Y' the row is skipped (132 rows in total)
            This matrix is used to solve the system (Rho_X'Y' * m - Rho_XY = 0)
        
        Parameters
        ----------
            data: pandas DataFrame
                of deltaG over reactions and substituents
        
        Returns    (m_matrix)
        ----------
            m_matrix : 2D np.array
                matrix A for the WLSQ
        
        Other
        ----------
            This function does not call any other function
            
            This function is CALLED by:
                - calc_init_rho
    """
    reactions = list(data.columns)
    num_reactions = len(reactions)
    m_matrix=np.zeros(num_reactions)
    
    for idx1,rxn1 in enumerate(reactions):
        for idx2,rxn2 in enumerate(reactions):
            if (rxn1==rxn2): continue
            pairwise = data[[rxn1, rxn2]].copy().dropna(axis=0,how='any')
            pairwise.columns = ['first', 'second']
            if (len(pairwise)<2):
                continue
            else:
                newline=np.zeros(num_reactions)
                slope = scipy.stats.mstats.theilslopes(pairwise[pairwise.columns[1]].values,                                                       pairwise[pairwise.columns[0]], alpha=0.5)[0]
                newline[idx1]= (-1)
                newline[idx2]= slope
                m_matrix = np.vstack((m_matrix,newline))
    m_matrix=np.delete(m_matrix, 0, 0)
    
    return(m_matrix)


def calc_init_rho(data, ref = 0):
    """ This function generates the INITIAL set of rhos (BEFORE the eventual Self-Consistency)
            WLSQ regression on the system Ax=b, where we fix the fisrt m to be 1 to avoid trivial solutions
            For this reason the b vector is not only zeros, but contains the first column of the matrix, which has been
            dropped from the A matrix
    
        Parameters
        ----------
            data: pandas DataFrame
                of deltaG over reactions and substituents
                
            ref = 0: float
                optiona, choice of reference reaction
        
        Returns    (dicrho, rhos)
        ----------
            dicrho: dictionary
                key:= XY, value:= rho
                
            rhos: 1D np.array
                contains the values of rhos
                
        Other
        ----------
            This function calls:
                - calc_m
    """
     
    slopesmatr = calc_m(data)
    
    A = np.delete(slopesmatr, ref, 1)
    
    b = -slopesmatr[:,ref]
    
    regr_rhos = np.linalg.lstsq(A , b , rcond=None)[0]
   
    rhos = np.insert(regr_rhos, ref, 1)
    
    dicrho = dict(zip(list(data.columns), rhos))
    
    return(dicrho, rhos)


def calc_sigmas(data, dicrho):
    """ This function generates the set of sigmas
    
        Parameters
        ----------
            data : pandas DataFrame
                of deltaG over reactions and substituents

            dicrhos :  dict
                key := reaction, value := rho
    
        Returns   (dicsigmas, sigmas)
        ----------
            dicsigmas :  dict
                key := string R_1 to R_4, value := sigma
        
            sigmas : 1D np.array
                set of the sigmas
                
        Others
        -----------
            This function does NOT CALL anything else
            
            This function is CALLED by:
                - Hammett_data
    """
    
    heading = list(data.columns)
    rhos = np.vectorize(dicrho.get)(heading)
        
    sigmas = (data / rhos).mean(axis=1).values
    
    dicsigmas = dict(zip(data.index.values, sigmas))
    
    return(dicsigmas, sigmas)


def fixrho(data, dicsigmas):
    ''' This function calculates the final set of rhos and E0. The dic of sigmas is necessary
        in case of ML but does not hurt otherwise
    
        Parameters
        ----------
            df: pd.DataFrame
                reshaped df with the training data (columns: 4label, DeltaG(XY) )
                
            dicsigmas: dictionary
                key:= R1_R2_R3_R4, value:= sigma, 
                contains a values for ALL the sigmas, test set obtained via huber alphas
    
        
        Returns    (dicr2, dice2)
        ----------
            dicr2: dictionary
                key:= XY, value:= rho
        
            dice2: dictionary
                key:= XY, value:= E0
                
    
        Other
        ----------
            No other functions are called here
     
    '''
    
    
    
    sigmastrain = pd.merge(data, pd.DataFrame.from_dict(dicsigmas, orient='index'),                   left_index=True, right_index=True)[0].values
    
    rho2 = []
    e2 = []

    for rxn in list(data.columns):
        varx = sigmastrain
        vary = data[rxn].values
        mask = ~np.isnan(varx) & ~np.isnan(vary)
        slope, intercept = scipy.stats.mstats.theilslopes(vary[mask],  varx[mask])[0:2]
        rho2.append(slope)
        e2.append(intercept)

    dicr2 = dict(zip(list(data.columns), rho2))
    dice2 = dict(zip(list(data.columns), e2))
    
    return(dicr2, dice2, np.array(rho2), np.array(e2))


def calc_params(data, ref = 0):
    '''This function calculates the rhos, sigmas and E0 of the Hammett regression. 
        
        Parameters
        ----------
            data : Pandas DataFrame
            One row for each molecule with the corresponding 12 Activation Energies (reshaped)
         
        Returns      (dicrho, dicx0, dicsigmas, rhos, sigmas)
        -----------
            dicnewrho: dictionary
                key:= reaction, value:= rho value

            dicx0: dictionary
                key:= reaction, value:= E0 value

            dicsigmas: dictionary
                key:= R1_R2_R3_R4, value:= sigma

            rhos: 1D np.array
                of the same 12 rhos (same found in dictrhos)

            sigmas: 1D np.array
                of the sigmas (same found in dicsigmas)
         
        
        Other
        ----------
            This function calls:
                - calc_init_rho
                - calc_sigmas
    '''
    x0 = data.median(axis=0).values
    dicrho, rhos = calc_init_rho(data, ref)
    dicsigmas, sigmas = calc_sigmas(data - x0, dicrho) 
    dicx0 = dict(zip(list(data.columns), x0))
    
    dicnewrho, dicE0, newrho, E0 = fixrho(data, dicsigmas)
    
    return(dicnewrho, dicE0, dicsigmas, newrho, E0, sigmas)


def evaluate(data, rho, sigma, E0):
    '''
        This function creates a prediction dataframe with the same shape as the input one filled with values obtained by the Enhanced-Hammett algorithm

        Parameters
        ------------
            data: pd.DataFrame with the initial data, used in order to get the structure

            rho, sigma, E0: 1D np.array with the corresponding values, they must have the same order as rows and columns


        Returns
        -----------
            hammpred: pd.DataFrame with the prediction


    '''
    prediction = pd.DataFrame().reindex_like(data).fillna(1.0) 
    hammpred = prediction.multiply(rho, axis=1).multiply(sigma, axis=0) + E0
    return hammpred


def plot_correlation(data, prediction):
    '''
        This function plots the correlation between true and predicted values

        Parameters
        -------------
            data: pd.DataFrame with original values

            prediction: pd.DataFrame with prediction

        Returns
        -------------
            plot

        Other
        -------------
            Data and Prediction must have the same shape, it is recommended to generate Prediction using
            the "evaluate" function
    '''
    
    true = data.values.ravel()
    pred = prediction.values.ravel()
    
    fig, ax = plt.subplots()
    ax.plot(true, pred, 'o')

    #make the correlation plot square
    plt.gca().set_aspect('equal', adjustable='box')

    #axis aspect
    ax.plot([0, 1], [0, 1], ls="--", c=".3", zorder=0,transform=ax.transAxes)
    ax.set_xlabel('True Value', fontsize=15)
    ax.set_ylabel('Prediction', fontsize=15)
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
    ax.tick_params(axis='both', labelsize=13)
    plt.show()



def get_experimental_data_Hudson_1962(path='dataset_Hudson_1962.csv'):
    '''
        This function generates the dataframe with the example experimental data [1]
        
        Parameters
        --------------
        path: str   path to the 'dataset.csv' file
        
        Returns
        --------------
        data: pd.DataFrame with the parsed data
        
        
        [1]  Hudson, R. F.; Klopman, G. J. Chem. Soc. 1962, 1062–1067
        
    '''
    data = pd.read_csv(path, index_col=0)
    return(data)

