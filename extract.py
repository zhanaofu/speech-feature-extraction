import numpy as np
import pandas as pd
import argparse


def mutual_information(matrix):
    """Calculating mutual information from tabulated data"""
    total = matrix.sum()
    flatM = matrix.ravel()
    margin0 = matrix.sum(axis=0)
    margin1 = matrix.sum(axis=1)
    divisor = np.outer(margin1,margin0).ravel()
    indices = flatM.nonzero()
    return np.sum(flatM[indices]/total*np.log2(flatM[indices]*total/divisor[indices]))

def mutual_arrays(a1,a2):
    """Calculating mutual information from 2 arrays"""
    u1 = list(set(a1))
    u2 = list(set(a2))
    ar = np.zeros([len(u1),len(u2)])
    for i in range(len(a1)):
        ar[u1.index(a1[i]),u2.index(a2[i])]+=1
    return mutual_information(ar)


def mutual_x_y(cm,matrix, counts, feature, output_grouping = True, ratio = True,printi = False): 
    """Tabulating data based on a binary feature and the outcome from the full confusion matrix. 
    If output_grouping is True, the outcome is also grouped based on the feature"""
    if output_grouping:
        indices0 = []
        indices1 = []
        for i,v in enumerate(feature):
            if v:
                indices1.append(i)
            else:
                indices0.append(i)

        matrix_xy = np.array([[matrix[np.ix_(indices0,indices0)].sum(),
                          matrix[np.ix_(indices1,indices0)].sum()],
                         [matrix[np.ix_(indices0,indices1)].sum(),
                          matrix[np.ix_(indices1,indices1)].sum()]])
    else:
        a0 = matrix[:,feature].sum(axis=1)
        a1 = matrix[:,np.invert(feature)].sum(axis=1)
        matrix_xy = np.vstack([a0,a1]).transpose()
        
    result = mutual_information(matrix_xy)
    
    if printi:
        maximum = mutual_information(np.array([[counts[indices0].sum(),0],
                                                     [0,counts[indices1].sum()]]))
        result = [result,result/maximum,maximum]
    elif ratio:
        cm.I_x=np.append(cm.I_x,mutual_information(np.array([[counts[indices0].sum(),0],
                                                     [0,counts[indices1].sum()]])))
        result = result/cm.I_x[-1]
    
    return result


def mutual_xi_xj(counts, f1, f2):
    """Tabulating the input based on two features"""
    ind1 = []
    for u in np.unique(f1):
        ind1.append(np.where(f1 == u))
    ind2 = []
    for u in np.unique(f2):
        ind2.append(np.where(f2 == u))   
    xl = len(ind1)
    yl = len(ind2)
    matrix_xx = np.zeros((xl,yl),dtype=int)

    for i in range(yl):
        for j in range(xl):
            ind = list(set(ind1[j][0])&set(ind2[i][0]))
            matrix_xx[j,i]=counts[ind].sum()
    return mutual_information(matrix_xx)

def mergef(a,b):
    """Merging two binary features a and b into a multi-valent feature."""
    c = 0
    uni = {}
    t = []
    for i in zip(a,b):
        if i not in uni:
            uni[i]=c
            c+=1
        t.append(uni[i])
    return t

def J(cm,k,method,merge = False,filt = True):
    """Selecting features from a comfusion matrix.
    cm: A confusion_matrix object.
    k: The number of features to be selected.
    merge: Whether to merge the selected features for the calculation of mutual information between selected features and the candidate feature.
    filt: Whether to remove the non-contrasting features from the candidate feature set.
    """
    try:
        selected = cm.results[method.__name__]
        merged = cm.merged_temp

    except:
        selected = np.array([],dtype=int)
        merged = np.ones(cm.n, dtype = int)
 
    indices = cm.order
    
    for i in range(len(selected),k):
        best = 0
        j_max = 0

        for j in range(len(indices)):
            f = indices[j]
            I_xx = 0
            I_xx_Y = 0
            j_x = cm.MI_xy[f]
            if merge:
                cm.features[cm.n_features]=merged
                j_x += method(cm,(f,cm.n_features),I_xx,I_xx_Y,len(selected))
            else:
                for s in selected:
                    if f < s:       
                        a = f
                        b = s
                    else:
                        a = s
                        b = f
                    j_x += method(cm,(a,b),I_xx,I_xx_Y,len(selected),f)
            if j_x > j_max:
                j_max = j_x
                best = j
                
                
        # merge selected features

        merged = mergef(merged,cm.features[indices[best]])
        cm.merged_temp = merged
        
        selected = np.append(selected, indices[best])
        indices=np.delete(indices,best)
        if merge:
            # reset MI
            cm.MI_xx = {}
            cm.MI_xx_Y = {}


        if filt:
            lcs = []

            for u in np.unique(merged):
                tv = merged==u           
                if tv.sum()>1:
                    lcs.append(np.where(tv)[0])

            rs = []

            for x in range(len(indices)):
                for y in lcs:
                    
                    if (np.all(cm.features[indices[x]][y]) or np.all(1-(cm.features[indices[x]][y]))):
                        rs.append(x)
                        continue

            indices=np.delete(indices,rs)

    cm.order=indices
    return selected
    
def RCT(cm,p,I_xx,I_xx_Y,n,f):
    """To maximize Redundancy-Corrected Transmissibility"""
    if not p in cm.MI_xx:
        cm.MI_xx[p] = mutual_xi_xj(cm.x_counts,cm.features[p[0]],cm.features[p[1]])                  
    I_xx += cm.MI_xx[p]/cm.I_x[f]
    if I_xx:
        I_xx = I_xx/n

    return  -I_xx
    
class confusion_matrix:
    def __init__(self, df,base = 2):
        if 'csv' in df:
            self.df = pd.read_csv(df, index_col=0)
        else:
            self.df = df.copy()
        self.df.reindex(sorted(self.df.index), axis=0)
        self.df.reindex(sorted(self.df.columns), axis=1)
        
        self.matrix = self.df.to_numpy()
        self.x_counts = self.matrix.sum(axis=0)
        self.n = self.matrix.shape[0]
        self.ny = self.matrix.shape[1]
        self.py = self.matrix.sum(axis=1)/self.matrix.sum()
        self.results = {}
        self.selected_features = {}
        self.same_items= {}
        self.results_resampled = {}
        self.same_items_resampled = {}
        self.selected_features_resampled = {}
        self.base = base
        self.n_features = self.base**(self.n-1)-1
        
        self.MI_xy = np.array([])
        self.I_x = np.array([])
        self.MI_xx = {}
        # Uppercase Y means Y is the conditioning variable
        self.MI_xx_Y = {}
                
    def not_diff(self, method,resample = False):
        if resample:
            results = self.results_resampled[method]
            self.same_items_resampled[method]=[]
            for i in range(len(results)):
                j = results[i]
                f = self.features[j]
                if not i:
                    fs = f
                else:
                    fs = np.vstack([fs, f])
            for i in range(fs.shape[1]):
                for j in range(i+1,fs.shape[1]):
                    if (fs[:,i]==fs[:,j]).all():
                        self.same_items_resampled[method].append([self.df.index[i],self.df.index[j]])
        else:    
            results = self.results[method]
            self.same_items[method]=[]
            for i in range(len(results)):
                j = results[i]
                f = self.features[j]
                if not i:
                    fs = f
                else:
                    fs = np.vstack([fs, f])

            for i in range(fs.shape[1]):
                for j in range(i+1,fs.shape[1]):
                    if (fs[:,i]==fs[:,j]).all():
                        self.same_items[method].append([self.df.index[i],self.df.index[j]])


    def select(self, feature_lim = None, k = False, method = "mRMR",
               merge = False, output_grouping = False, stop=True,
              ratio = True, filt = False):
        
        self.features = []
        
        for i in range(0,self.n_features):
            f = np.array(list(np.base_repr(i+1,base=self.base)),dtype=int)
            f = np.append(np.zeros(self.n-f.shape[0],dtype=int),f)
            self.features.append(f)
        # the extra feature is a placeholder for the merged feature during selection
        self.features.append(np.zeros(self.n,dtype = int))
        self.features = np.array(self.features)
        
        if self.MI_xy.size==0:     
            print('Computing I(x;y)')

            for i in range(self.n_features):
                self.MI_xy= np.append(self.MI_xy,mutual_x_y(self,self.matrix,self.x_counts,self.features[i],
                                                            ratio=ratio,output_grouping=output_grouping))

        self.order = np.arange(self.n_features, dtype='int')  
        
        if not k:
            k = int(np.ceil(np.log2(self.n)))
            self.same_items[method]=True
            print('Selecting features to differentiate all items with {}'.format(method))
            print('Number of features being selected: ',end='')
            while bool(self.same_items[method]):
                if stop & (k > self.n):
                    break
                print('{} '.format(k),end='')
                exec('self.results[method] = J(self,{},{},{},{})'.format(k,method,merge,filt))
                self.not_diff(method)
                k+=1
            self.selected_features[method] = np.array([self.features[j] for j in self.results[method]])
            print('')
        else:
            maxk = k
            k = int(np.ceil(np.log2(self.n)))
            self.same_items[method]=True
            print('Selecting {} features with {}'.format(maxk,method))
            print('Number of features being selected: ',end='')
            while bool(self.same_items[method]):
                print('{} '.format(k),end='')
                exec('self.results[method] = J(self,{},{},{},{})'.format(k,method,merge,filt))
                self.not_diff(method)
                k+=1
            self.order = np.arange(self.n_features, dtype='int')
            self.order = [o for o in self.order if o not in self.results[method]]        
            print('Selecting {} features with {}'.format(maxk,method))
            print('{} '.format(maxk),end='')
            exec('self.results[method] = J(self,{},{},{},{})'.format(maxk,method,merge,False))
            self.selected_features[method] = np.array([self.features[j] for j in self.results[method]])
        print('Completed!')

     
    def save_selected(self,out):
        with open(out,'w') as f:
            f.write(','.join(self.df.columns)+'\n')
            for i in range(self.selected_features['RCT'].shape[0]):
                f.write(','.join(map(str,self.selected_features['RCT'][i]))+'\n')
        with open(out.replace('.csv','.txt'),'w') as f:
            for i in range(self.selected_features['RCT'].shape[0]):
                feature = self.selected_features['RCT'][i]
                groups = ['' for i in range(self.base)]

                for j in range(len(feature)):
                    groups[feature[j]]+=self.df.columns[j]+' '
                f.write('| '.join(groups)+'\n')

        
        print('Results saved to {} and {}.'.format(out,out.replace('.csv','.txt')))
        
    def show_terms(self):
        df = pd.DataFrame(columns=['method','no','I_xk_Y','I_xk_xj','I_xk_xj/S',
                                   'I_xk_xj_Y','I_xk_xj_Y/S','I_xk_y_Xj'])
           
        for method,results in self.results.items(): 
            n = len(results)
            for k in range(0,n):
                f = results[k]
                
                if k == 0:
                    df=df.append({'method':method,'no':1,'I_xk_Y':self.MI_xy[f],'I_xk_xj':'NA',
                                 'I_xk_xj/S':'NA','I_xk_xj_Y':'NA','I_xk_xj_Y/S':'NA',
                                 'I_xk_y_Xj':'NA'},ignore_index=True)
                else:
                    I_xk_xj = 0
                    I_xk_xj_Y = 0
                    I_xk_y_Xj = 0
                    for j in range(0,k):
                        s = results[j]
                        if f < s:       
                            a = f
                            b = s
                        else:
                            a = s
                            b = f
                        try:
                            I_xk_xj += self.MI_xx[(a,b)]
                        except KeyError:
                            self.MI_xx[(a,b)] = mutual_xi_xj(self.x_counts,self.features[a],self.features[b])
                            I_xk_xj += self.MI_xx[(a,b)]
                        I_xk_y_Xj += self.MI_xy[f] - self.MI_xx[(a,b)] + self.MI_xx_Y[(a,b)]
                        
                    df=df.append({'method':method,'no':k+1,'I_xk_Y':self.MI_xy[f],'I_xk_xj':I_xk_xj,
                                 'I_xk_xj/S':I_xk_xj/k,'I_xk_xj_Y':I_xk_xj_Y,
                                  'I_xk_xj_Y/S':I_xk_xj_Y/k,
                                 'I_xk_y_Xj':I_xk_y_Xj},ignore_index=True)
        
        with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
            print(df)
           

def balance(df,ratio,subtract):
    
    dfc = df.copy()
    if subtract:
        dfc = dfc - dfc.min().min() +1
    n = dfc.shape[0]
    
    for i in range(n):

        col = np.copy(dfc.iloc[:,i])
        c = col.sum()
        col[i] = 0
        dfc.iloc[:,i]=col/col.sum()
    np.fill_diagonal(dfc.values,ratio)
    # print(dfc)
    return dfc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extracting binary features from a confusion matrix.'
    )

    parser.add_argument(
        'file', type=str, nargs='?',
        help='The path to the .csv file containing the confusion matrix. The first column and the first row are the names of the phonemes.'
    )
    parser.add_argument(
        '--out', type=str,
        help='''The path to the output .csv file. Default: (name of the data file)_features.csv.'''
    )

    parser.add_argument(
        '--preprocessing', type=str, default='p', choices=['p','skip'],
        help='''How to preprocess the data before feature extraction. Options: 'p' - calculating the probabilities of errors from each input phoneme (the probability of errors in each COLUMN adds up to 1 ); 'skip' - no preprocessing. Default: 'p'.'''
    )

    parser.add_argument(
        '--ratio', type = float, default=1,
        help='''If 'p' is selected as the preprocessing method, this argument assigns the ratio between numbers of correct mappings (the numbers in diagonal cells) and (the numbers in off-diagonal cells) incorrect mappings for each INPUT phoneme. Requires a number. Default: 1.'''
    )

    parser.add_argument(
        '--subtract', type = str, default=True, choices=['True','False'],
        help='''If 'p' is selected as the preprocessing method, this argument specifies whether to subtract the smallest value in the matrix minus 1 from the whole matrix. Options: True or False. Default: True.'''
    )
 
    args = parser.parse_args()
    if args.out is None:
        args.out = args.file.replace('.csv','_features.csv')
    df = pd.read_csv(args.file,index_col=0)
    if args.preprocessing=='p':
        f = balance(df,args.ratio,args.subtract=='True')
    cm = confusion_matrix(f)
    cm.select(method = 'RCT',output_grouping=True, ratio = True, filt = True)
    cm.save_selected(args.out)