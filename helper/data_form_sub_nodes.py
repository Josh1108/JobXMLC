import json
import random
from scipy.sparse import csr_matrix,save_npz
with open("./../dumps/mycareersfuture.json") as f:
    data = json.load(f)
# add job category 
print(type(data['jobs']))
random.shuffle(data['jobs'])

count_max =len(data['jobs'])- len(data['jobs'])/10
count_max_cat = count_max
f = open("../../data/COLING-sub-nodes/trn_X_req-role.txt",'w')
g = open("../../data/COLING-sub-nodes/trn_X_job-req.txt",'w')
h = open("../../data/COLING-sub-nodes/trn_X_job-category-list.txt",'w')
l = open("../../data/COLING-sub-nodes/trn_X_job-title.txt",'w')
q = open("../../data/COLING-sub-nodes/trn_X_seniority.txt",'w')
f_t = open("../../data/COLING-sub-nodes/tst_X_req-role.txt",'w')
g_t = open("../../data/COLING-sub-nodes/tst_X_job-req.txt",'w')
l_t = open("../../data/COLING-sub-nodes/tst_X_job-title.txt",'w')
q_t = open("../../data/COLING-sub-nodes/tst_X_seniority.txt",'w')

job_cat = set()

for item in data['jobs']:
    if item['job_category']!=[]:
        for it in item['job_category']:
            job_cat.add(it)
    if count_max>0: 
        if item['requirements_and_role']!="":
            f.write(item['requirements_and_role']+'\n')
        else:
            f.write('<empty>\n')
        if item['job_requirements']!="":
            g.write(item['job_requirements']+'\n')
        else:
            g.write('<empty>\n')
        if item['seniority']:
            q.write(item['seniority']+'\n')
        else:
            q.write('<empty>\n')
        if item['job_title']:
            l.write(item['job_title']+'\n')
        else:
            l.write('<empty>\n')
        count_max-=1
    else:
        if item['requirements_and_role']!="":
            f_t.write(item['requirements_and_role']+'\n')
        else:
            f_t.write('<empty>\n')
        if item['job_requirements']!="":
            g_t.write(item['job_requirements']+'\n')
        else:
            g_t.write('<empty>\n')
        if item['seniority']:
            q_t.write(item['seniority']+'\n')
        else:
            q_t.write('<empty>\n')
        if item['job_title']:
            l_t.write(item['job_title']+'\n')
        else:
            l_t.write('<empty>\n')
        count_max-=1
        
job_cat = list(job_cat)
dicti = {x:i for i,x in enumerate(job_cat)}
for item in job_cat:
    h.write(item+'\n')

matrix_train=[]
matrix_test =[]
for item in data['jobs']:
    if count_max_cat>0:
        lis_temp =[]
        for it in item['job_category']:
            lis_temp.append(dicti[it])
        matrix_train+=lis_temp
    else:
        lis_temp =[]
        for it in item['job_category']:
            lis_temp += dicti[it]
        matrix_test+=lis_temp
matrix_train=csr_matrix(matrix_train)
matrix_test=csr_matrix(matrix_test)
save_npz('../../data/COLING-sub-nodes/trn_X_job-category',matrix_train)
save_npz('../../data/COLING-sub-nodes/tst_X_job-category',matrix_test)



    

