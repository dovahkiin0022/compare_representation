import re
import numpy as np
from pymatgen import core as mg
import pickle
import pandas as pd

gfa_dataset_file = 'gfa_dataset.txt'
z_row_column_file = 'Z_row_column.txt'
element_property_file = 'element_property.txt'
common_path = "Files_from_GTDL_paper/{}" 
RC = pickle.load(open(common_path.format(z_row_column_file), 'rb')) 
new_index=[int(i[4]) for i in RC]#new order 
Z_row_column = pickle.load(open(common_path.format(z_row_column_file), 'rb'))
[property_name_list,property_list,element_name,_]=pickle.load(open(common_path.format(element_property_file), 'rb'))


periodic_table_file = '1d_orders/periodic_table.csv'
periodic_df = pd.read_csv(periodic_table_file)
atomic_number_order = periodic_df['Symbol'].values[:103] #only the first 103 elements

alternate_orders_file = '1d_orders/alternate_orders.pkl'
with open(alternate_orders_file,'rb') as fid:
    alternate_order_dict = pickle.load(fid)
pettifor_order = alternate_order_dict['pettifor']
modified_pettifor_order = alternate_order_dict['modified_pettifor']

def special_formatting(comp):
  """take pymatgen compositions and does string formatting"""
  comp_d = comp.get_el_amt_dict()
  denom = np.sum(list(comp_d.values()))
  string = ''
  for k in comp_d.keys():
    string += k + '$_{' + '{}'.format(round(comp_d[k]/denom,3)) + '}$'
  return string

def PTR(comp,property_list = property_list,element_name = element_name,RC=RC):#PTR psuedoimage using special formula
    #i0='Mo.5Nb.5'
    #i=i0.split(' ')[0]
    i = special_formatting(comp)
    X= [[[0.0 for ai in range(18)]for aj in range(9)] for ak in range(1) ]  
    tx1_element=re.findall('[A-Z][a-z]?', i)#[B, Fe, P,No]
    tx2_temp=re.findall('[0-9.]+', i)#[$_{[50]}$, ] [50 30 20]
    tx2_value=[float(re.findall('[0-9.]+', i_tx2)[0]) for i_tx2 in tx2_temp]
    for j in range(len(tx2_value)):
        index=int(property_list[element_name.index(tx1_element[j])][1])#atomic number
        xi=int(RC[index-1][1])#row num
        xj=int(RC[index-1][2])#col num
        X[0][xi-1][xj-1]=tx2_value[j]

    #properties at the first row, from 5th to 8th column for hardness
    X = np.array(X)
    return X

def get_1d_features(comp,order='atomic', return_elements = False):
  """

  Args:
      comp (pymatgen composition): _description_
      order (str, optional): specifies the order in which elements should appear in the output feature array. Allowed values are 'atomic', 'pettifor' and 'mod_pettifor'.. Defaults to 'atomic'.
      return_elements (bool, optional): Specifies whether the element order is returned in the output. Defaults to False.

  Returns:
      A 1D feature array or a tuple containing feature array and element list
  """
  if order == 'atomic':
      el_list = atomic_number_order
  elif order == 'pettifor':
       el_list = pettifor_order
  elif order == 'mod_pettifor':
       el_list = modified_pettifor_order
  arr = np.zeros((1,len(el_list)))
  for (el,v) in comp.get_el_amt_dict().items():
        ind = np.where(np.array(el_list) == el)
        arr[:,ind] = v
  arr /= arr.sum(axis=1)
  if return_elements:
      return arr, el_list
  else:
       return arr



