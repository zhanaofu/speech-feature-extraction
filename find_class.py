import numpy as np
import pandas as pd
import argparse


def read_features(file):
    df = pd.read_csv(file)
    if isinstance(df.iloc[1,0], str):
        df.set_index(df.columns[0])
        df = df.iloc[:,1:]
    phonemes = df.columns
    features = df.values

    return phonemes,features

def read_classes(files):
    df = pd.read_csv('/Users/f/Documents/GitHub/feature_extraction/pbase_eng.csv')
    classes = {'rule':[],'distribution':[]}
    for _, row in df.iterrows():
        row_cls = str(row['classes'])
        if row_cls!='nan':
            row_cls = row_cls.split('][')
            for i in range(len(row_cls)):
                row_cls[i]=row_cls[i].replace('[','').replace(']','')
                
                if (row['Type'] == 'Rule') & (row_cls[i] not in classes['rule']):
                    classes['rule'].append(row_cls[i])
                elif (row['Type'] == 'Distribution') & (row_cls[i] not in classes['distribution']):
                    classes['distribution'].append(row_cls[i])
    return classes


def samei(matrix, v):
    '''For every given phoneme, check if it has value '1' for all features in the matrix.'''
    tv = np.array([True]*matrix.shape[1])
    for i in range(matrix.shape[0]):
        tv = tv & (matrix[i] == v)
    return tv

def elim(ma,class_ind,target):
    """In a set of features that are shared by (not necessarily defines) the target nutural class, find the smallest subset of features that still DEFINE the same set of phonemes."""
    matrix = np.copy(ma[class_ind,:])
    rown = matrix.shape[0]
    sme = rown
    maxr = 2 ** rown
    
    for i in range(rown):
        if matrix[i,target[0]]==0:
            matrix[i]=1-matrix[i]
    
    initiali = samei(matrix,1)

    inisum = initiali.sum()
    sumt = inisum
    best = np.array([True]*rown)
    
    for i in range(maxr):
        find = np.array(list(np.base_repr(i,base=2)),dtype=int)
        find = np.append(np.zeros(rown-find.shape[0],dtype=int),find).astype(bool)
        tot = samei(matrix[find],1).sum()

        if tot <= sumt:
            sumt = tot
            fnn = find.sum()
            
            if fnn<=sme:
                best = find
                sme = fnn
    return best, sumt-len(target)


class natclass:
    def __init__(self, phonemes, features):
        self.phonemes = phonemes
        self.features = features
        self.n_features = features.shape[0]
        self.n_phonemes = features.shape[1]
        maxi = int(''.join(['1']*self.n_features),2)
        self.feature_dic = {maxi:[i for i in range(self.n_phonemes)]}    
        self.mat = np.ones(features.shape).astype(int)
        # two classes
        self.gs = [np.nonzero(self.phonemes)[0],np.nonzero(self.phonemes-1)[0]]
        
    def show_common(self,file):
        rs = []
        for g in self.gs:
            if len(g)>1:
                r = []
                fs = self.features[:,g]
            
                for i in range(fs.shape[0]):
                    f = fs[i,:]
                    if np.ptp(f)==0: #ptp = Peak-to-peak (maximum - minimum) 
                        r.append([i,f[0]])

                if len(r)>0:
                    ind = np.array([True]*self.n_phonemes)
                    for p in r:
                        ind = ind&(self.features[p[0],:]==p[1])
                    
                    file.write('{}\n'.format(list(phonemes[ind])))
                    best,diffp = elim(self.features,[iit[0] for iit in r],g)
                    
                    r = [r[i] for i in range(len(r)) if best[i]]
                    
                    r.append(diffp)
                    file.write('{}\n'.format(r))

                    rs.append(r)
        if len(rs)==1:
            rs = rs[0]
        elif len(rs)==2:
            if (rs[0][-1] < rs[1][-1]) or ((rs[0][-1] == rs[1][-1]) & (len(rs[0])<=len(rs[1]))):
                rs = rs[0] 
            else:
                rs = rs[1]

        return rs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extracting binary features from a confusion matrix.'
    )

    parser.add_argument(
        'feature_file', type=str, nargs='?',
        help='The path to the .csv file containing the feature matrix. The first column and the first row are the names of the phonemes.'
    )
    parser.add_argument(
        'pattern_file', type=str,
        help='''The path to the .csv file containing the p-base patterns with classes added as the last column.'''
    )

    parser.add_argument(
        '--out', type=str,
        help='''The path to the output .csv file. Default: (name of the feature file)_classes.csv.'''
    )
 
    args = parser.parse_args()
    if args.out is None:
        args.out = args.feature_file.replace('.csv','_classes.txt')
    phonemes,features = read_features(args.feature_file)
    classes = read_classes(args.pattern_file)

    with open(args.out,'w') as f:
        for k,v in classes.items():
            f.write('==========================\n')
            f.write('==========================\n')
            f.write('Natural classes in patterns from {}.\n'.format(k))
            f.write('==========================\n')
            f.write('==========================\n')
            class_bool = []
            for i in v:
                class_bool.append([])
                for l in phonemes:
                    class_bool[-1].append(l in i)
            class_bool = np.array(class_bool).astype(int)

            sc = 0 # Number of successful matches between the real natural classes and the classes defined by the features
            diff = 0 # Average number of phoneme difference between the real natural classes and the classes defined by the feature
            fn = 0 # Average minimal number of features needed for the successful matches
            t= 0 # Natural classes with at least one shared feature

            for ci in range(class_bool.shape[0]):
                f.write('Natural class {}\n'.format(ci))
                f.write('Phonemes in the natural class: {} \n'.format(list(phonemes[np.nonzero(class_bool[ci,:])])))
                f.write('Phonemes not in the natural class: {} \n'.format(list(phonemes[np.nonzero(class_bool[ci,:]-1)])))
                f.write('----------\n')
                f.write('Closest class defined by the features:\n')
                c = class_bool[ci,:]
                n = natclass(c,features).show_common(f)
                if len(n)>0:
                    diff+= n[-1]
                    t+=1
                    if n[-1]==0:
                        fn += len(n)-1
                        sc+=1
                
                f.write('----------\n')
                f.write('----------\n')
            f.write('==========================\nSummary\n==========================\n')
            f.write('Natural classes with at least one shared feature: {}\n'.format(t))
            f.write('Average number of phoneme difference between the real natural classes and the classes defined by the feature: {}\n'.format(diff/t))
            f.write('Number of successful matches between the real natural classes and the classes defined by the features: {}\n'.format(sc))
            f.write('Average minimal number of features needed for the successful matches: {}\n\n\n\n\n\n'.format(fn/sc))