from astropy.table import Table, column
import math
import numpy as np
import pandas as pd

class ArcLookup():
    
    def __init__(self,dr8id):
        self.dr8id = dr8id
        
    def arcs(self,lookup_table,return_mask=False):
        arc_mask = lookup_table['gxyName'] == self.dr8id
        if return_mask == True:
            return arc_mask
        else:
            return lookup_table[arc_mask].sort_values('arc_length',
                                                      ascending=False)
        
    def chirality(self,lookup_array):
        return lookup_array[lookup_array[:,0] == self.dr8id][:,1][0]
    
    def less_arcs(self,lookup_table,chi=None,L=None,P=None,m=None,
                  return_mask=False):
        arcs = self.arcs(lookup_table)
        # only keep 'correct' chiralities:
        chi_lookup = {None: np.full(len(arcs),True,bool),
                      0: (arcs['pitch_angle'] < 0).as_matrix(),
                      1: (arcs['pitch_angle'] > 0).as_matrix()}
        chi_mask = chi_lookup[chi]
        # only keep 'long enough' arcs:
        if L == None:
            L_mask = np.full(len(arcs),True,bool)
        else:
            L_mask = (arcs['arc_length'] > L).as_matrix()
        # only keep 'ok P' arcs:
        if P == None:
            P_mask = np.full(len(arcs),True,bool)
        else:
            P_mask = ((arcs['pitch_angle'].abs() >= P[0]) & 
                      (arcs['pitch_angle'].abs() <= P[1])).as_matrix()
        # combine all of the masks:    
        combined_mask = (chi_mask) & (L_mask) & (P_mask)
        # Finally, only keep < m arcs:
        if m != None:
            N_true = 0
            for i, val in enumerate(combined_mask):
                N_true += int(val)
                combined_mask[i] = False if N_true > m else combined_mask[i]
                
        if return_mask == True:
            return combined_mask
        else:
            return arcs[combined_mask]
        
    def calculate_P(self,lookup_table,chi=None,L=None,P=None,m=None,
                    weighted_average=True):
        
        arcs = self.less_arcs(lookup_table,chi,L,P,m)
        N = len(arcs)
        if N == 0:
            return 0, 0
        else:
            weights = arcs['arc_length'].as_matrix() if weighted_average is True else None
            P = np.average(arcs['pitch_angle'],weights=weights)
            return N, P
    
    def arc_list(self,lookup_table,chi=None,L=None,P=None,m=None):
        arcs = self.less_arcs(lookup_table,chi,L,P,m)
        P = arcs['pitch_angle'].as_matrix().tolist()
        L = arcs['arc_length'].as_matrix().tolist()
        intensity = arcs['mean_intensity'].as_matrix().tolist()
        N_pixels = arcs['num_pixels'].as_matrix().tolist()
        N_arcs = len(P)
        fractional_length = [l/L[0] for l in L]
        arcs_list = [self.dr8id,N_arcs,P,L,intensity,N_pixels,fractional_length]
        return arcs_list
      
      
class ArcStats():
    
    def __init__(self,arc_table):
        self.arc_table = arc_table
        self.N_rows = len(arc_table)
        self.setup_array = np.zeros(self.N_rows)
        
    def value_to_array(self,value=0,variable_name='i'):
        N_rows = self.N_rows
        if (isinstance(value,int)) or (isinstance(value,float)):
            return np.full(N_rows,value)
        else:
            if len(value) != N_rows:
                raise ValueError('Length of variable {} does not'
                                 'match number of rows'.format(variable_name))
            else:
                return value
        
    def N_arcs_L(self,px=55):
        px_array = self.value_to_array(px)
        N = self.setup_array
        for r in range(self.N_rows):
            L_r = np.array(self.arc_table['L'][r])
            N[r] = (L_r > px_array[r]).sum()
        return N
    
    def N_arcs_Lf(self,f=0.5):
        f_array = self.value_to_array(f)
        N = self.setup_array
        for r in range(self.N_rows):
            L_r = np.array(self.arc_table['L_f'][r])
            N[r] = (L_r > f_array[r]).sum()
        return N
    
    def L_total(self,px=55):
        px_array = self.value_to_array(px)
        L = self.setup_array
        for r in range(self.N_rows):
            L_r = np.array(self.arc_table['L'][r])
            ok_L = L_r > px_array[r]
            L[r] = L_r[ok_L].sum()
        return L
    
    def P_average(self,px=55,weighted_average=True,absolute=True):
        px_array = self.value_to_array(px)
        P = self.setup_array
        for r in range(self.N_rows):
            L_r = np.array(self.arc_table['L'][r])
            P_r = np.array(self.arc_table['P'][r])
            ok_L = L_r > px_array[r]
            if weighted_average is True:
                weights = np.array(self.arc_table['L_f'][r])[ok_L]
            else:
                weights = None
            if ok_L.sum() == 0:
                P[r] = 0
            else:
                P[r] = np.average(P_r[ok_L],weights=weights)
        if absolute is True:
            return np.absolute(P)
        else:
            return P
        
    def N_weighted_average(self,px=0):
        px_array = self.value_to_array(px)
        N_wtd_avg = self.setup_array
        N_rows = self.N_rows
        for r in range(N_rows):
            L_r = np.array(self.arc_table['L'][r])
            Lf_r = np.array(self.arc_table['L_f'][r])
            ok_L = L_r > px_array[r]
            if ok_L.sum() > 0:
                N_wtd_avg[r] = (Lf_r[ok_L]).sum()
            else:
                N_wtd_avg[r] = 0
        return N_wtd_avg

      
def measure_P_values(ids,L=None,P=None,m=None):
    P_array = np.empty((len(ids),3))
    P_array[:,0] = ids
    if (type(m) == int) | (m == None):
        m = np.full(len(ids),m)
    for i, id_ in enumerate(ids):
        i_arcs = arc_lookup(id_)
        chi = i_arcs.chirality(chi_array)
        N_arcs, P_mean = i_arcs.calculate_P(sparcfire_g_arcs,
                                            chi,L,P,m[i])
        P_array[i,1:] = N_arcs, np.abs(P_mean)
    return Table(P_array,names=('id','N','P'),
                 dtype=[int,int,float])