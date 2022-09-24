from f3_create_epochs import *

epochs = create_epoches()


def artifact_removal3(data,p):
    AR_rem={}
    m2={}
    for i in data:
        m2[i]=np.zeros(np.shape(data[i])[2], dtype=bool)
        for j in range(np.shape(data[i])[0]):
            a=(np.max(data[i][j],axis=0))-(np.min(data[i][j],axis=0))
            z=((a-np.mean(a))/np.std(a))
            avglatlist = np.arange(1, a.shape[0] + 1)
            m=np.abs(z) > 2.5
            #mask.append(m)
            m2[i]=np.logical_or(m2[i],m)

            #a=(np.var(data[i][j],axis=0))
            a=np.sum(np.abs(data[i][j]),axis=0)
            z=((a-np.mean(a))/np.std(a))
            m=np.abs(z) > 2.5
            #mask.append(m)
            m2[i]=np.logical_or(m2[i],m)

        AR_rem[i] = np.delete(data[i],avglatlist[m2[i]]-1,2)
        mini=10**10

        for ele in AR_rem:
            if np.shape(AR_rem[ele])[2]<mini:
                mini=np.shape(AR_rem[ele])[2]
    return AR_rem,mini,m2
