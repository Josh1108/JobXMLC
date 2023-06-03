import numpy as np


def othermetrics(true_labels, pred_labels):
    true_lbls_t = true_labels.toarray()
    pred_lbls_t = pred_labels.toarray()
    kvals = [5,10,30,50,100]
    ndcg_k=[]
    rr = []
    y_true_list = true_lbls_t
    y_pred = pred_lbls_t
    for i in range(0,len(y_true_list)):
        y_t = y_true_list[i]
        y_t_len = len(y_t)
        y_p_index = np.flip(np.argsort(y_pred[i]),0)
        for i in range(0,len(y_p_index)):
            if y_p_index[i] in y_t:
                rr.append(1/float(i+1))
                break
        idcg = np.sum([1.0/np.log2(x+2) for x in range(0,y_t_len)])
        r = []
        ndcg = []
        for k in kvals:
            correct = len(np.intersect1d(y_t,y_p_index[0:k]))
            r.append(correct/float(y_t_len))
            dcg = 0
            for i in range(0,k):
                if y_p_index[i] in y_t:
                    dcg = dcg + 1.0/np.log2(i+2)
            ndcg.append(dcg/idcg)
        ndcg_k.append(ndcg)
    ndcg = np.mean(ndcg_k,axis=0)*100
    return np.mean(rr), ndcg

# @nb.njit(cache=True)
def _recall(true_labels_indices, true_labels_indptr,
            pred_labels_data, pred_labels_indices, pred_labels_indptr, k):
    fracs = []
    for i in range(len(true_labels_indptr) - 1):
        _true_labels = true_labels_indices[true_labels_indptr[i]: true_labels_indptr[i + 1]]
        _data = pred_labels_data[pred_labels_indptr[i]: pred_labels_indptr[i + 1]]
        _indices = pred_labels_indices[pred_labels_indptr[i]: pred_labels_indptr[i + 1]]
        top_inds = np.argsort(_data)[::-1][:k]
        _pred_labels = _indices[top_inds]
            

        if(len(_true_labels) > 0):
            fracs.append(len(set(_pred_labels).intersection(
                set(_true_labels))) / len(_true_labels))
    return np.mean(np.array(fracs, dtype=np.float32))


def _precision(true_labels_indices, true_labels_indptr,
            pred_labels_data, pred_labels_indices, pred_labels_indptr, k):
    fracs =[]        
    for i in range(len(true_labels_indptr) - 1):
        _true_labels = true_labels_indices[true_labels_indptr[i]: true_labels_indptr[i + 1]]
        _data = pred_labels_data[pred_labels_indptr[i]: pred_labels_indptr[i + 1]]
        _indices = pred_labels_indices[pred_labels_indptr[i]: pred_labels_indptr[i + 1]]
        top_inds = np.argsort(_data)[::-1][:k]
        _pred_labels = _indices[top_inds]

    
        if(len(_pred_labels) > 0):
            fracs.append(len(set(_pred_labels).intersection(
                set(_true_labels))) / len(_pred_labels))
    return np.mean(np.array(fracs, dtype=np.float32))

def precision(true_labels,pred_labels,k):
    return _precision(true_labels.indices.astype(np.int64), true_labels.indptr,
                   pred_labels.data, pred_labels.indices.astype(np.int64), pred_labels.indptr, k)

def recall(true_labels, pred_labels, k):
    return _recall(true_labels.indices.astype(np.int64), true_labels.indptr,
                   pred_labels.data, pred_labels.indices.astype(np.int64), pred_labels.indptr, k)