import numpy as np
import pandas as pd
import argparse


# python extract.py cm_perception.csv -resample False


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
    """Tabulating data based on a feature and the outcome from the full confusion matrix. 
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


def mutual_xi_xj_Y(cm, indices_i1, indices_j1): 
    """Calculating conditional mutual information"""
    mi = 0
    for r in range(cm.ny):
        mi+= cm.py[r]*mutual_xi_xj(cm.matrix[r,], indices_i1, indices_j1)
    return mi

def mergef(a,b):
    """Merging features a and b"""
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
    
def rel(cm,p,I_xx,I_xx_Y,n,f):
    """To maximize MI_xy - (1/k) * MI_xx (Peng et al., 2005)"""
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
        
        
    def show_selected(self, method, matrix = False):
        results = self.results[method]
        print('Printing {} selected features:'.format(len(results)))
        print('[-feature] | [+feature] | [MI, MI/maximum, maximum MI]')
        for k in results:
            feature = self.features[k]
            groups = ['' for i in range(self.base)]

            for i in range(len(feature)):
                groups[feature[i]]+=self.df.columns[i]

            groups.append(str(mutual_x_y(self,self.matrix,self.x_counts,feature,printi=True)))
            
            print(' | '.join(groups))
     
        
        
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
                        try:
                            I_xk_xj_Y += self.MI_xx_Y[(a,b)]
                        except KeyError:
                            self.MI_xx_Y[(a,b)] = mutual_xi_xj_Y(self,self.features[a],self.features[b])
                            I_xk_xj_Y += self.MI_xx_Y[(a,b)]
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
           

def balance(df,error = 5000, diag = 5000,subtract =False):
    
    dfc = df.copy()
    if subtract:
        dfc = dfc - dfc.min().min() +1
    n = dfc.shape[0]
    
    for i in range(n):

        col = np.copy(dfc.iloc[:,i])
        c = col.sum()
        col[i] = 0
        dfc.iloc[:,i]=0
        pcol = col/col.sum()

        for j in np.random.choice(n,error,p=pcol):
            dfc.iloc[j,i]+=1

        dfc.iloc[i,i]=diag
    return dfc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extracting binary features from a confusion matrix.'
    )

    parser.add_argument(
        'file', type=str, nargs='?',
        help='The confusion matrix in a csv file'
    )
    parser.add_argument(
        '-resample', type=bool, default=True,
        help='Resample the data?'
    )

    args = parser.parse_args()

    df = pd.read_csv(args.file,index_col=0)
    if args.resample:
        f = balance(df)
    cm = confusion_matrix(df)
    cm.select(method = 'rel',output_grouping=True, ratio = True, filt = True)
    cm.show_selected('rel')