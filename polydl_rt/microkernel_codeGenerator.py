#!/usr/bin/env python
# coding: utf-8


# ### How to Run file?
# ### python3 filename vec_step_M vec_step_M vec_step_M Ti Tj Tk

# In[1]:

import sys


# In[2]:


output = "output.cpp"


# ### New File

# In[3]:


def reset():
    f = open(output, "w")
    f.write('')
    f.close()


# ## Open / Close Function 

# In[4]:


def open_function(step_M,step_N,step_K):
    f = open(output, "a+") 
    f.write('void polydl_lib_matmul_f32_i_%d_j_%d_k_%d_fma( long long int M, long long int N, long long int K,long long int A_stride, long long int B_stride, long long int C_stride, float *A, float *B, float *C) {\n' %(step_M,step_N,step_K))
    f.close()
    
def close_function():
    f = open(output, "a+") 
    f.write('} \n')
    f.close()    


# ## Declarations

# In[5]:


def declaration(step_M,step_N,step_K, Ti, Tj, Tk):
    f = open(output, "a+")
    f.write('__m512 vec_C;\n')
    f.write('__m512 vec_B;\n')
    f.write('__m512 vec_A;\n')
    
    for variable in ['C','A']:
        f.write('__m512 ')
        for i in range (1,step_M):
            f.write('vec_' + variable +str(i))
            if(i==step_M-1):
                f.write(';\n')
            else:
                f.write(', ')
    
    f.write('int i, j, k;\n')
    
    f.write('long long M_full, N_full, K_full;\n')
    if(Ti and Tj and Tk):
        f.write('M_full = (M / Ti) * Ti ;\n' )
        f.write('N_full = (N / Tj) * Tj ;\n' )
        f.write('K_full = (K / Tk) * Tk ;\n' )
    else:
        f.write('M_full = (M / %d) * %d ;\n' %(step_M,step_M))
        f.write('N_full = (N / %d) * %d ;\n' %(step_N,step_N))
    
    f.close()
        


# ### # Define functions

# In[6]:


def defineMinMax():
    f = open(output, "a+")
    f.write('#define min(X, Y) (((X) < (Y)) ? (X) : (Y))\n')
    f.write('#define max(X, Y) (((X) > (Y)) ? (X) : (Y))\n')
    f.close()

def GemmBlock(Ti, Tj, Tk):
    f = open(output, "a+")
    f.write('#ifndef Ti \n#define Ti %d \n#endif\n' % Ti)
    f.write('#ifndef Tj \n#define Tj %d \n#endif\n' %Tj)
    f.write('#ifndef Tk \n#define Tk %d \n#endif\n' %Tk)
    f.close()


# ## For Loops

# In[7]:


def loopClosingbracket():
    f = open(output, "a+")
    f.write('}\n')
    f.close()
def loopOver(step_M,step_N,step_K,Ti,Tj,Tk):
    f = open(output, "a+")
    
    # Tiled Loops
    if(Ti and Tj and Tk):
        f.write('int it,jt,kt;\n')
        f.write('for (it = 0; i < M_full; it+=Ti) {\n')
        f.write('for (jt = 0; jt < N_full; jt+=Tj) {\n')
        f.write('for (kt = 0; kt < K_full; kt+=Tk) {\n')
        M_Conditional = 'min(M_full, it+Ti)'
        N_Conditional = 'min(N_full, jt+Tj)'
        K_Conditional = 'min(K_full, kt+Tk)'
        i_start = 'it'
        j_start = 'jt'
        k_start = 'kt'
    else:
        M_Conditional = 'M_full'
        N_Conditional = 'N_full'
        K_Conditional = 'K'
        i_start = '0'
        j_start = '0'
        k_start = '0'
        
    # i Loop
    f.write('for (i = %s; i < %s; i += %d) {\n' % (i_start,M_Conditional,step_M))
    
    # j Loop
    f.write('for (j = %s; j < %s; j += %d) {\n' % (j_start,N_Conditional, step_N))
    f.write('vec_C = _mm512_load_ps((__m512*)&C[i*C_stride + j]);\n')
    
    for i in range(1, step_M):
        f.write('vec_C%d = _mm512_load_ps((__m512*)&C[(i + %d)*C_stride + j]);\n' % (i, i))
    
    # k Loop
    f.write('for (k = %s; k < %s; k += %d) {\n' % (k_start,K_Conditional,step_K))
    
    f.write('vec_B = _mm512_load_ps((__m512*)&B[k*B_stride + j]);\n')
    
    f.write('vec_A = _mm512_set1_ps(A[i*A_stride + k]);\n')
    for i in range(1, step_M):
        f.write('vec_A%d = _mm512_set1_ps(A[(i + %d)*A_stride + k]);\n' % (i,i))
    
    f.write('vec_C = _mm512_fmadd_ps(vec_A, vec_B, vec_C);\n')
    for i in range(1, step_M):
        f.write('vec_C%d = _mm512_fmadd_ps(vec_A%d, vec_B, vec_C%d);\n' % (i,i, i))
    
    
    f.close()
    
    # k Loop
    loopClosingbracket()
    
    f = open(output, "a+")
    f.write('_mm512_store_ps((__m512*)&C[i*C_stride + j], vec_C);\n')
    for i in range(1, step_M):
        f.write('_mm512_store_ps((__m512*)&C[(i + %d)*C_stride + j], vec_C%d);\n' % (i,i))
    f.close()
    
    # j Loop
    loopClosingbracket()
    # i Loop
    loopClosingbracket()
    
    # Closing Brackets for Tiled loops
    if(Ti and Tj and Tk):
        loopClosingbracket()
        loopClosingbracket()
        loopClosingbracket()


# In[10]:

if len(sys.argv):
    step_M=int(sys.argv[1])
    step_N=int(sys.argv[2])
    step_K=int(sys.argv[3])
    Ti = int(sys.argv[4])
    Tj = int(sys.argv[5])
    Tk = int(sys.argv[6])
else:
    step_M=8
    step_N=16
    step_K=1
    Ti = 16
    Tj = 16
    Tk = 16

reset()

if (Ti and Tj):
    GemmBlock(Ti, Tj, Tk)
    defineMinMax()
    
open_function(step_M,step_N,step_K)

declaration(step_M,step_N,step_K,Ti, Tj, Tk)

loopOver(step_M,step_N,step_K, Ti, Tj, Tk)


close_function()