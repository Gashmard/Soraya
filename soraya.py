
import numpy as np
import pandas as pd
import mendeleev
from mendeleev import *
from mendeleev import element
from mendeleev.fetch import fetch_ionization_energies
import chemparse
import pymatgen
from pymatgen.core.composition import Composition

# Function to read and convert file to DataFrame
def read_input_file(input_file):
    # Determine the file format based on the file extension
    file_extension = input_file.split('.')[-1].lower()
    if file_extension == 'csv':
        # If the input file is in CSV format, read it directly as a DataFrame
        dataframe_input_file = pd.read_csv(input_file)
    elif file_extension in ['xlsx', 'xls']:
        # If the input file is in Excel format (XLSX or XLS), read it as a DataFrame
        dataframe_input_file = pd.read_excel(input_file)
    elif file_extension == 'json':
        # If the input file is in JSON format, read it as a DataFrame
        dataframe_input_file = pd.read_json(input_file)
    else:
        print(f"Unsupported file format: '{file_extension}'.", 
        "Please convert the input file format to one of the following formats:'csv', 'xlsx', 'xls' or 'json'.")
    return dataframe_input_file

def data_preprocessing(DataSet_for_Generation_322Features):
    # Determine the column name of the formula
    target_column = 'formula'
    # Checking the formula column to have the not a number or the same NaN
    if DataSet_for_Generation_322Features[target_column].isna().any():
        print(f"Column '{target_column}' contains \033[1m\033[97m\033[7m NaN \033[0m values.")

    List_Compound_Not_in_Periodic_Table=[]
    for index, row in DataSet_for_Generation_322Features.iterrows():
        # Creating a dictionary of elements
        Element_dict=chemparse.parse_formula(row["formula"]) 
        Element_dict_in_periodic_table=chemparse.parse_formula('Lr1Md1Fm1Es1Cf1Bk1Cm1Ac1Ra1Fr1Rn1At1Po1Pm1Xe1Kr1Ar1Ne1No1He1Al1Ag1Zn1Cd1Sn1Tl1Ti1Rh1In1Bi1Pb1Te1Hg1Ge1Th1Se1Mo1Ga1Sb1S1B1La1Nb1F1I1Pd1O1N1H1V1Au1Cr1Zr1Be1Fe1Si1Mg1Li1Er1Mn1As1Ru1Os1Ce1Cu1Gd1Ni1Sc1Ca1Sr1Y1Ho1Ta1C1U1Pt1Yb1Re1Am1P1W1Lu1Ba1Hf1Nd1K1Na1Cl1Pr1Co1Rb1Eu1Cs1Ir1Tc1Sm1Dy1Tm1Br1Tb1Pa1Np1Pu1')
        for k,v in Element_dict.items():
            if k not in Element_dict_in_periodic_table:
                List_Compound_Not_in_Periodic_Table.append(row["formula"])
    if len(List_Compound_Not_in_Periodic_Table) != 0:
        print('\033[1m\033[97m\033[7m Contains problematic compounds \033[0m', 
               'These compounds contain elements not found in the periodic table=', List_Compound_Not_in_Periodic_Table)

    for index, row in DataSet_for_Generation_322Features.iterrows():
        try:
            Elem = row['formula']
            comp = Composition(Elem)
        except Exception as e:
            print("This compound is problematic, please correct", Elem,".  ", e)

def Generation_322_Features(DataSet_for_Generation_322Features):
  import warnings
  warnings.filterwarnings('ignore')
  # Number_of_Elements
  Number_of_Elements=[]
  for index, row in DataSet_for_Generation_322Features.iterrows():
      comp = Composition(row['formula'])
      Number=len(comp) 
      Number_of_Elements.append(Number)   
  DataSet_for_Generation_322Features['Number_Elements']= Number_of_Elements 

  #  Sum_of_Subscript
  Sum_Subscript_List=[]
  for index, row in DataSet_for_Generation_322Features.iterrows():   
      comp = Composition(row['formula'])
      Sum_Subscript_atoms =comp.num_atoms
      Sum_Subscript_List.append(Sum_Subscript_atoms)
  DataSet_for_Generation_322Features['Sum_Subscript']= Sum_Subscript_List

  # Electrongativity
  # Based_fraction
  total_electronegativ_Based_fraction_mean=[]
  total_electronegativ_Based_fraction_median=[]
  total_electronegativ_Based_fraction_variance=[]
  total_electronegativ_Based_fraction_max=[]
  total_electronegativ_Based_fraction_min=[]
  total_electronegativ_Based_fraction_range=[]  
  total_electronegativ_Based_fraction_std=[] # std= Standard deviation  
  total_electronegativ_Based_fraction_Ave_dev =[]   # Ave_dev=Average_deviation 
  # Based_Elemental
  total_electronegativ_Based_Elemental_mean=[]
  total_electronegativ_Based_Elemental_median=[]
  total_electronegativ_Based_Elemental_variance=[]
  total_electronegativ_Based_Elemental_max=[]
  total_electronegativ_Based_Elemental_min=[]
  total_electronegativ_Based_Elemental_range=[] 
  total_electronegativ_Based_Elemental_std=[] # std= Standard deviation
  total_electronegativ_Based_Elemental_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Subscript
  total_electronegativ_Based_Subscript_mean=[]
  total_electronegativ_Based_Subscript_median=[]
  total_electronegativ_Based_Subscript_variance=[]
  total_electronegativ_Based_Subscript_max=[]
  total_electronegativ_Based_Subscript_min=[]
  total_electronegativ_Based_Subscript_range=[]
  total_electronegativ_Based_Subscript_std=[] # std= Standard deviation  
  total_electronegativ_Based_Subscript_Ave_dev =[]   # Ave_dev=Average_deviation  

  # pettifor_number=Pettifor_numb 
  # Based_fraction
  total_pettifor_number_Based_fraction_mean=[]
  total_pettifor_number_Based_fraction_median=[]
  total_pettifor_number_Based_fraction_variance=[]
  total_pettifor_number_Based_fraction_max=[]
  total_pettifor_number_Based_fraction_min=[]
  total_pettifor_number_Based_fraction_range=[] 
  total_pettifor_number_Based_fraction_std=[] 
  total_pettifor_number_Based_fraction_Ave_dev =[] 
  # Based_Elemental
  total_pettifor_number_Based_Elemental_mean=[]
  total_pettifor_number_Based_Elemental_median=[]
  total_pettifor_number_Based_Elemental_variance=[]
  total_pettifor_number_Based_Elemental_max=[]
  total_pettifor_number_Based_Elemental_min=[]
  total_pettifor_number_Based_Elemental_range=[]  
  total_pettifor_number_Based_Elemental_std=[] 
  total_pettifor_number_Based_Elemental_Ave_dev =[] 
  # Based_Subscript
  total_pettifor_number_Based_Subscript_mean=[]
  total_pettifor_number_Based_Subscript_median=[]
  total_pettifor_number_Based_Subscript_variance=[]
  total_pettifor_number_Based_Subscript_max=[]
  total_pettifor_number_Based_Subscript_min=[]
  total_pettifor_number_Based_Subscript_range=[] 
  total_pettifor_number_Based_Subscript_std=[] # std= Standard deviation 
  total_pettifor_number_Based_Subscript_Ave_dev =[]   # Ave_dev=Average_deviation 

  # van der waals radius=VanderwaalsR 
  # Based_Elemental
  total_VanderwaalsR_Based_Elemental_mean=[]
  total_VanderwaalsR_Based_Elemental_median=[]
  total_VanderwaalsR_Based_Elemental_variance=[]
  total_VanderwaalsR_Based_Elemental_max=[]
  total_VanderwaalsR_Based_Elemental_min=[]
  total_VanderwaalsR_Based_Elemental_range=[]  
  total_VanderwaalsR_Based_Elemental_std=[] # std= Standard deviation  
  total_VanderwaalsR_Based_Elemental_Ave_dev =[]   # Ave_dev=Average_deviation 

  # period number
  # Based_Elemental
  total_period_number_Based_Elemental_mean=[]
  total_period_number_Based_Elemental_median=[]
  total_period_number_Based_Elemental_variance=[]
  total_period_number_Based_Elemental_max=[]
  total_period_number_Based_Elemental_min=[]
  total_period_number_Based_Elemental_range=[] 
  total_period_number_Based_Elemental_std=[] # std= Standard deviationpredictions = model_final.predict(x_test)
  total_period_number_Based_Elemental_Ave_dev =[]  # Ave_dev=Average_deviation 

  # group number
  # Based_Elemental
  total_group_number_Based_Elemental_mean=[]
  total_group_number_Based_Elemental_median=[]
  total_group_number_Based_Elemental_variance=[]
  total_group_number_Based_Elemental_max=[]
  total_group_number_Based_Elemental_min=[]
  total_group_number_Based_Elemental_range=[] 
  total_group_number_Based_Elemental_std=[] # std= Standard deviationpredictions = model_final.predict(x_test)
  total_group_number_Based_Elemental_Ave_dev =[]  # Ave_dev=Average_deviation 

  # heat_of_formation=heat_formation
  # Based_fraction
  total_heat_formation_Based_fraction_mean=[]
  total_heat_formation_Based_fraction_median=[]
  total_heat_formation_Based_fraction_variance=[]
  total_heat_formation_Based_fraction_max=[]
  total_heat_formation_Based_fraction_min=[]
  total_heat_formation_Based_fraction_range=[]  
  total_heat_formation_Based_fraction_std=[] # std= Standard deviation 
  total_heat_formation_Based_fraction_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Elemental
  total_heat_formation_Based_Elemental_mean=[]
  total_heat_formation_Based_Elemental_median=[]
  total_heat_formation_Based_Elemental_variance=[]
  total_heat_formation_Based_Elemental_max=[]
  total_heat_formation_Based_Elemental_min=[]
  total_heat_formation_Based_Elemental_range=[]  
  total_heat_formation_Based_Elemental_std=[] # std= Standard deviation  
  total_heat_formation_Based_Elemental_Ave_dev =[]   # Ave_dev=Average_deviation 
  # Based_Subscript
  total_heat_formation_Based_Subscript_mean=[]
  total_heat_formation_Based_Subscript_median=[]
  total_heat_formation_Based_Subscript_variance=[]
  total_heat_formation_Based_Subscript_max=[]
  total_heat_formation_Based_Subscript_min=[]
  total_heat_formation_Based_Subscript_range=[]  
  total_heat_formation_Based_Subscript_std=[] # std= Standard deviation  
  total_heat_formation_Based_Subscript_Ave_dev =[]   # Ave_dev=Average_deviation 

  # unpaired_electrons
  # Based_fraction
  total_unpaired_electron_Based_fraction_mean=[]
  total_unpaired_electron_Based_fraction_median=[]
  total_unpaired_electron_Based_fraction_variance=[]
  total_unpaired_electron_Based_fraction_max=[]
  total_unpaired_electron_Based_fraction_min=[]
  total_unpaired_electron_Based_fraction_range=[] 
  total_unpaired_electron_Based_fraction_std=[] # std= Standard deviation 
  total_unpaired_electron_Based_fraction_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Elemental
  total_unpaired_electron_Based_Elemental_mean=[]
  total_unpaired_electron_Based_Elemental_median=[]
  total_unpaired_electron_Based_Elemental_variance=[]
  total_unpaired_electron_Based_Elemental_max=[]
  total_unpaired_electron_Based_Elemental_min=[]
  total_unpaired_electron_Based_Elemental_range=[]  
  total_unpaired_electron_Based_Elemental_std=[] # std= Standard deviation  
  total_unpaired_electron_Based_Elemental_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Subscript
  total_unpaired_electron_Based_Subscript_mean=[]
  total_unpaired_electron_Based_Subscript_median=[]
  total_unpaired_electron_Based_Subscript_variance=[]
  total_unpaired_electron_Based_Subscript_max=[]
  total_unpaired_electron_Based_Subscript_min=[]
  total_unpaired_electron_Based_Subscript_range=[]  
  total_unpaired_electron_Based_Subscript_std=[] # std= Standard deviation 
  total_unpaired_electron_Based_Subscript_Ave_dev =[]   # Ave_dev=Average_deviation  

  #  number electron valence= Number_Elec_Valence
  # Based_fraction
  total_Number_Elec_Valence_Based_fraction_mean=[]
  total_Number_Elec_Valence_Based_fraction_median=[]
  total_Number_Elec_Valence_Based_fraction_variance=[]
  total_Number_Elec_Valence_Based_fraction_max=[]
  total_Number_Elec_Valence_Based_fraction_min=[]
  total_Number_Elec_Valence_Based_fraction_range=[]  
  total_Number_Elec_Valence_Based_fraction_std=[] # std= Standard deviation  
  total_Number_Elec_Valence_Based_fraction_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Elemental
  total_Number_Elec_Valence_Based_Elemental_mean=[]
  total_Number_Elec_Valence_Based_Elemental_median=[]
  total_Number_Elec_Valence_Based_Elemental_variance=[]
  total_Number_Elec_Valence_Based_Elemental_max=[]
  total_Number_Elec_Valence_Based_Elemental_min=[]
  total_Number_Elec_Valence_Based_Elemental_range=[]  
  total_Number_Elec_Valence_Based_Elemental_std=[] # std= Standard deviation  
  total_Number_Elec_Valence_Based_Elemental_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Subscript
  total_Number_Elec_Valence_Based_Subscript_mean=[]
  total_Number_Elec_Valence_Based_Subscript_median=[]
  total_Number_Elec_Valence_Based_Subscript_variance=[]
  total_Number_Elec_Valence_Based_Subscript_max=[]
  total_Number_Elec_Valence_Based_Subscript_min=[]
  total_Number_Elec_Valence_Based_Subscript_range=[]  
  total_Number_Elec_Valence_Based_Subscript_std=[] # std= Standard deviation  
  total_Number_Elec_Valence_Based_Subscript_Ave_dev =[]   # Ave_dev=Average_deviation 

  # spin_only_magnetic_moment=MagMoment
  # Based_fraction
  total_MagMoment_Based_fraction_mean=[]
  total_MagMoment_Based_fraction_median=[]
  total_MagMoment_Based_fraction_variance=[]
  total_MagMoment_Based_fraction_max=[]
  total_MagMoment_Based_fraction_min=[]
  total_MagMoment_Based_fraction_range=[]  
  total_MagMoment_Based_fraction_std=[] # std= Standard deviation  
  total_MagMoment_Based_fraction_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Elemental
  total_MagMoment_Based_Elemental_mean=[]
  total_MagMoment_Based_Elemental_median=[]
  total_MagMoment_Based_Elemental_variance=[]
  total_MagMoment_Based_Elemental_max=[]
  total_MagMoment_Based_Elemental_min=[]
  total_MagMoment_Based_Elemental_range=[]  
  total_MagMoment_Based_Elemental_std=[] # std= Standard deviation  
  total_MagMoment_Based_Elemental_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Subscript
  total_MagMoment_Based_Subscript_mean=[]
  total_MagMoment_Based_Subscript_median=[]
  total_MagMoment_Based_Subscript_variance=[]
  total_MagMoment_Based_Subscript_max=[]
  total_MagMoment_Based_Subscript_min=[]
  total_MagMoment_Based_Subscript_range=[]  
  total_MagMoment_Based_Subscript_std=[] # std= Standard deviation  
  total_MagMoment_Based_Subscript_Ave_dev =[]   # Ave_dev=Average_deviation 

  #  dipole_polarizability
  # Based_fraction
  total_dipole_polarizability_Based_fraction_mean=[]
  total_dipole_polarizability_Based_fraction_median=[]
  total_dipole_polarizability_Based_fraction_variance=[]
  total_dipole_polarizability_Based_fraction_max=[]
  total_dipole_polarizability_Based_fraction_min=[]
  total_dipole_polarizability_Based_fraction_range=[]  
  total_dipole_polarizability_Based_fraction_std=[] # std= Standard deviation  
  total_dipole_polarizability_Based_fraction_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Elemental
  total_dipole_polarizability_Based_Elemental_mean=[]
  total_dipole_polarizability_Based_Elemental_median=[]
  total_dipole_polarizability_Based_Elemental_variance=[]
  total_dipole_polarizability_Based_Elemental_max=[]
  total_dipole_polarizability_Based_Elemental_min=[]
  total_dipole_polarizability_Based_Elemental_range=[]  
  total_dipole_polarizability_Based_Elemental_std=[] # std= Standard deviation  
  total_dipole_polarizability_Based_Elemental_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Subscript
  total_dipole_polarizability_Based_Subscript_mean=[]
  total_dipole_polarizability_Based_Subscript_median=[]
  total_dipole_polarizability_Based_Subscript_variance=[]
  total_dipole_polarizability_Based_Subscript_max=[]
  total_dipole_polarizability_Based_Subscript_min=[]
  total_dipole_polarizability_Based_Subscript_range=[]  
  total_dipole_polarizability_Based_Subscript_std=[] # std= Standard deviation  
  total_dipole_polarizability_Based_Subscript_Ave_dev =[]   # Ave_dev=Average_deviation 

  #  First Ionisation Energy= First_Ionis_Energy
  # Based_fraction
  total_First_Ionis_Energy_Based_fraction_mean=[]
  total_First_Ionis_Energy_Based_fraction_median=[]
  total_First_Ionis_Energy_Based_fraction_variance=[]
  total_First_Ionis_Energy_Based_fraction_max=[]
  total_First_Ionis_Energy_Based_fraction_min=[]
  total_First_Ionis_Energy_Based_fraction_range=[]  
  total_First_Ionis_Energy_Based_fraction_std=[] # std= Standard deviation  
  total_First_Ionis_Energy_Based_fraction_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Elemental
  total_First_Ionis_Energy_Based_Elemental_mean=[]
  total_First_Ionis_Energy_Based_Elemental_median=[]
  total_First_Ionis_Energy_Based_Elemental_variance=[]
  total_First_Ionis_Energy_Based_Elemental_max=[]
  total_First_Ionis_Energy_Based_Elemental_min=[]
  total_First_Ionis_Energy_Based_Elemental_range=[]  
  total_First_Ionis_Energy_Based_Elemental_std=[] # std= Standard deviation  
  total_First_Ionis_Energy_Based_Elemental_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Subscript
  total_First_Ionis_Energy_Based_Subscript_mean=[]
  total_First_Ionis_Energy_Based_Subscript_median=[]
  total_First_Ionis_Energy_Based_Subscript_variance=[]
  total_First_Ionis_Energy_Based_Subscript_max=[]
  total_First_Ionis_Energy_Based_Subscript_min=[]
  total_First_Ionis_Energy_Based_Subscript_range=[]  
  total_First_Ionis_Energy_Based_Subscript_std=[] # std= Standard deviation  
  total_First_Ionis_Energy_Based_Subscript_Ave_dev =[]   # Ave_dev=Average_deviation 

  #ElecAffinity
  Create_Features_dict_wiki = {'element':['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
  'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
  'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu',
  'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr',
  'Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg',
  'Cn','Nh','Fl','Mc','Lv','Ts','Og'],
  'Electron_affinity_Wiki_Kj_mol':['72.76','-48','59.63','-48','26.98','121.77','-6.8','140.97','328.16',
  '-116','52.86','-40','41.76','134.06','72.03','200.41','348.57','-96','48.38','2.37','18','7.28','50.91','65.21','-50',
  '14.78','63.89','111.65','119.23','-58','29.06','118.93','77.65','194.95','324.53','-96','46.88','5.02','29.6','41.80',
  '88.51','72.1','53','100.96','110.27','54.24','125.86','-68','37.04','107.29','101.05','190.16','295.15','-77',
  '45.5','13.95','53.79','55','10.53','9.4','12.45','15.63','11.2','13.22','12.67','33.96','32.61','30.1','99','-1.93',
  '23.04','17.18','31','78.76','5.82','103.99','150.90','205.04','222.74','-48','30.88','34.41','90.92','136','233.08'
  ,'-68','46.89','9.64','33.77','58.63','53.03','30.39','45.85','-48.33','9.93','27.17','-165.24','-97.31','-28.60',
  '33.96','93.91','-223.22','-30.04','0','0','0','0','0','0','0','151','0','66.6','0','35.3','74.9','165.9','5.4']}                               
  ElecAffinity_Wiki = pd.DataFrame(Create_Features_dict_wiki)  
  # Based_fraction
  total_ElecAffinity_Based_fraction_mean=[]
  total_ElecAffinity_Based_fraction_median=[]
  total_ElecAffinity_Based_fraction_variance=[]
  total_ElecAffinity_Based_fraction_max=[]
  total_ElecAffinity_Based_fraction_min=[]
  total_ElecAffinity_Based_fraction_range=[]  
  total_ElecAffinity_Based_fraction_std=[] # std= Standard deviation  
  total_ElecAffinity_Based_fraction_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Elemental
  total_ElecAffinity_Based_Elemental_mean=[]
  total_ElecAffinity_Based_Elemental_median=[]
  total_ElecAffinity_Based_Elemental_variance=[]
  total_ElecAffinity_Based_Elemental_max=[]
  total_ElecAffinity_Based_Elemental_min=[]
  total_ElecAffinity_Based_Elemental_range=[]  
  total_ElecAffinity_Based_Elemental_std=[] # std= Standard deviation 
  total_ElecAffinity_Based_Elemental_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Subscript
  total_ElecAffinity_Based_Subscript_mean=[]
  total_ElecAffinity_Based_Subscript_median=[]
  total_ElecAffinity_Based_Subscript_variance=[]
  total_ElecAffinity_Based_Subscript_max=[]
  total_ElecAffinity_Based_Subscript_min=[]
  total_ElecAffinity_Based_Subscript_range=[]  
  total_ElecAffinity_Based_Subscript_std=[] # std= Standard deviation  
  total_ElecAffinity_Based_Subscript_Ave_dev =[]   # Ave_dev=Average_deviation  

  # thermal_conductivity= Thermal_conduct
  Create_Features_dict_Thermal_Comd = {'element':['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
  'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
  'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu',
  'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr',
  'Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg',
  'Cn','Nh','Fl','Mc','Lv','Ts','Og'],
  'Thermal_Conductivity_W_mk':['0.1805','0.1513','85','190','27','140','0.02583','0.02658','0.0277',
  '0.0491','140','160','235','150','0.236','0.205','0.0089','0.01772','100','200','16','22','31','94','7.8',
  '80','100','91','400','120','29','60','50','0.52','0.12','0.00943','58','35','17','23',
  '54','139','51','120','150','72','430','97','82','67','24','3','0.449','0.00565',
  '36','18','13','11','13','17','15','13','14','11','11','11','16','15','17','39',
  '16','23','57','170','48','88','150','72','320','8.3','46','35','8','20','2'
  ,'0.00361','15','19','12','54','47','27','6','6','10','10','10','10','10',
  '10','10','10','10','23','58','19','0.01','0.01','0.01','0.01','0.01','0.01','0.01','0.01','0.01','0.01','0.01','0.0023']}                               
  Thermal_Conductivity = pd.DataFrame(Create_Features_dict_Thermal_Comd)
  # Based_fraction
  total_Thermal_conduct_Based_fraction_mean=[]
  total_Thermal_conduct_Based_fraction_median=[]
  total_Thermal_conduct_Based_fraction_variance=[]
  total_Thermal_conduct_Based_fraction_max=[]
  total_Thermal_conduct_Based_fraction_min=[]
  total_Thermal_conduct_Based_fraction_range=[]  
  total_Thermal_conduct_Based_fraction_std=[] # std= Standard deviation  
  total_Thermal_conduct_Based_fraction_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Elemental
  total_Thermal_conduct_Based_Elemental_mean=[]
  total_Thermal_conduct_Based_Elemental_median=[]
  total_Thermal_conduct_Based_Elemental_variance=[]
  total_Thermal_conduct_Based_Elemental_max=[]
  total_Thermal_conduct_Based_Elemental_min=[]
  total_Thermal_conduct_Based_Elemental_range=[]  
  total_Thermal_conduct_Based_Elemental_std=[] # std= Standard deviation  
  total_Thermal_conduct_Based_Elemental_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Subscript
  total_Thermal_conduct_Based_Subscript_mean=[]
  total_Thermal_conduct_Based_Subscript_median=[]
  total_Thermal_conduct_Based_Subscript_variance=[]
  total_Thermal_conduct_Based_Subscript_max=[]
  total_Thermal_conduct_Based_Subscript_min=[]
  total_Thermal_conduct_Based_Subscript_range=[]  
  total_Thermal_conduct_Based_Subscript_std=[] # std= Standard deviation  
  total_Thermal_conduct_Based_Subscript_Ave_dev =[]   # Ave_dev=Average_deviation 

  # Electrical_conductivity= Electric_Conduct 
  Create_Features_dict_Electric_Cond = {'element':['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
  'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
  'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu',
  'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr',
  'Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg',
  'Cn','Nh','Fl','Mc','Lv','Ts','Og'],
  'Elec_Conductivity_MS_m':['0.00012','0.00012','11','25','0.000000001','0.1','0.00012','0.00012','0.00012',
  '0.0001','21','23','38','0.001','10','0.000000000000000000001','0.00000001','0.00012','14','29','1.8','2.5','5','7.9','0.62',
  '10','17','14','59','17','7.1','0.002','3.3','0.00012','0.00000000000000001','0.00012','8.3','7.7','1.8','2.4',
  '6.7','20','5','14','23','10','62','14','12','9.1','2.5','0.01','0.0000000000001','0.00012',
  '5','2.9','1.6','1.4','1.4','1.6','1.3','1.1','1.1','0.77','0.83','1.1','1.1','1.2','1.4','3.6',
  '1.8','3.3','7.7','20','5.6','12','21','9.4','45','1.0','6.7','4.8','0.77','2.3','0.00012'
  ,'0.00012','0.00012','1.0','0.00012','6.7','5.6','3.6','0.83','0.67','0.00012','0.00012','0.00012','0.00012','0.00012',
  '0.00012','0.00012','0.00012','0.00012','0.00012','0.00012','0.00012','0.00012','0.00012','0.00012','0.00012','0.00012',
  '0.00012','0.00012','0.00012','0.00012','0.00012','0.00012','0.00012']}                               
  Electrical_Conductivity = pd.DataFrame(Create_Features_dict_Electric_Cond) 
  # Based_fraction
  total_Electric_Conduct_Based_fraction_mean=[]
  total_Electric_Conduct_Based_fraction_median=[]
  total_Electric_Conduct_Based_fraction_variance=[]
  total_Electric_Conduct_Based_fraction_max=[]
  total_Electric_Conduct_Based_fraction_min=[]
  total_Electric_Conduct_Based_fraction_range=[]  
  total_Electric_Conduct_Based_fraction_std=[] # std= Standard deviation  
  total_Electric_Conduct_Based_fraction_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Elemental
  total_Electric_Conduct_Based_Elemental_mean=[]
  total_Electric_Conduct_Based_Elemental_median=[]
  total_Electric_Conduct_Based_Elemental_variance=[]
  total_Electric_Conduct_Based_Elemental_max=[]
  total_Electric_Conduct_Based_Elemental_min=[]
  total_Electric_Conduct_Based_Elemental_range=[]  
  total_Electric_Conduct_Based_Elemental_std=[] # std= Standard deviation  
  total_Electric_Conduct_Based_Elemental_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Subscript
  total_Electric_Conduct_Based_Subscript_mean=[]
  total_Electric_Conduct_Based_Subscript_median=[]
  total_Electric_Conduct_Based_Subscript_variance=[]
  total_Electric_Conduct_Based_Subscript_max=[]
  total_Electric_Conduct_Based_Subscript_min=[]
  total_Electric_Conduct_Based_Subscript_range=[]  
  total_Electric_Conduct_Based_Subscript_std=[] # std= Standard deviation  
  total_Electric_Conduct_Based_Subscript_Ave_dev =[]   # Ave_dev=Average_deviation 

  # Specific heat= Specific_heat 
  Create_Features_dict_Specific_heat = {'element':['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
  'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
  'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu',
  'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr',
  'Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg',
  'Cn','Nh','Fl','Mc','Lv','Ts','Og'],
  'Specific_heat_J_KgK':['14300','5193.1','3570','1820','1030','710','1040','919','824',
  '1030','1230','1020','904','710','769.7','705','478.2','520.33','757','631','567','520','489','448','479',
  '449','421','445','384.4','388','371','321.4','328','321.2','947.3','248.05','364','300','298','278',
  '265','251','63','238','240','240','235','230','233','217','207','201','429','158.32',
  '242','205','195','192','193','190','180','196','182','240','182','167','165','168','160','154',
  '154','144','140','132','137','130','131','133','129.1','139.5','129','127','122','100',
  '90','93.65','90','92','120','118','99.1','116',
  '100','100','100','100','100','100','100','100','100','100','100','100','100','100','100'
  ,'100','100','100','100','100','100','100','100','100','100','100']}                               
  Specific_heat = pd.DataFrame(Create_Features_dict_Specific_heat)     
  # Based_fraction
  total_Specific_heat_Based_fraction_mean=[]
  total_Specific_heat_Based_fraction_median=[]
  total_Specific_heat_Based_fraction_variance=[]
  total_Specific_heat_Based_fraction_max=[]
  total_Specific_heat_Based_fraction_min=[]
  total_Specific_heat_Based_fraction_range=[]  
  total_Specific_heat_Based_fraction_std=[] # std= Standard deviation  
  total_Specific_heat_Based_fraction_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Elemental
  total_Specific_heat_Based_Elemental_mean=[]
  total_Specific_heat_Based_Elemental_median=[]
  total_Specific_heat_Based_Elemental_variance=[]
  total_Specific_heat_Based_Elemental_max=[]
  total_Specific_heat_Based_Elemental_min=[]
  total_Specific_heat_Based_Elemental_range=[]  
  total_Specific_heat_Based_Elemental_std=[] # std= Standard deviation  
  total_Specific_heat_Based_Elemental_Ave_dev =[]   # Ave_dev=Average_deviation  
  # Based_Subscript
  total_Specific_heat_Based_Subscript_mean=[]
  total_Specific_heat_Based_Subscript_median=[]
  total_Specific_heat_Based_Subscript_variance=[]
  total_Specific_heat_Based_Subscript_max=[]
  total_Specific_heat_Based_Subscript_min=[]
  total_Specific_heat_Based_Subscript_range=[]  
  total_Specific_heat_Based_Subscript_std=[] # std= Standard deviation  
  total_Specific_heat_Based_Subscript_Ave_dev =[]   # Ave_dev=Average_deviation 

  # Ionic_Radius
  Ionic_Radius_Element_periodic_tabel = {'element':['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
  'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
  'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu',
  'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr',
  'Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg',
  'Cn','Nh','Fl','Mc','Lv','Ts','Og'],
  'Ionic_Radius_Based_Angestrom':['1.48','0.4','0.76','0.27','0.11','0.15','1.46','1.38','1.3',
  '0.4','1.02','0.72','0.39','0.26','0.17','1.84','1.81','0.4','1.38','1.12','0.745','0.67','0.46','0.615','0.83',
  '0.78','0.745','0.69','0.65','0.6','0.47','0.39','0.335','1.98','1.96','0.4','1.56','1.21','0.9','0.72',
  '0.72','0.65','0.645','0.68','0.665','0.86','0.94','0.87','0.8','0.69','0.76','2.21','2.2','0.4',
  '1.74','1.35','1.1','1.01','0.99','1.109','0.97','1.079','1.01','1','0.923','0.912','0.901','0.89','0.88','0.868'
  ,'0.861','0.83','0.69','0.62','0.63','0.49','0.625','0.625','0.85','0.69','1.5','1.19','1.03','0.94',
  '0.62','0.4','1.8','1.7','1.12','1.09','1.01','1','0.98','0.96','1.09','0.95','0.93','0.92','0.4','0.4','0.4','1.1','0.4',
  '0.4','0.4','0.4','0.4','0.4','0.4','0.4','0.4','0.4','0.4','0.4','0.4','0.4','0.4','0.4']}                               
  df_Ionic_Radius_Element_periodic_tabel = pd.DataFrame(Ionic_Radius_Element_periodic_tabel)
  # Based_Elemental
  total_Ionic_Radius_Based_Elemental_mean=[]
  total_Ionic_Radius_Based_Elemental_median=[]
  total_Ionic_Radius_Based_Elemental_variance=[]
  total_Ionic_Radius_Based_Elemental_max=[]
  total_Ionic_Radius_Based_Elemental_min=[]
  total_Ionic_Radius_Based_Elemental_range=[]  
  total_Ionic_Radius_Based_Elemental_std=[] # std= Standard deviation  
  total_Ionic_Radius_Based_Elemental_Ave_dev =[]   # Ave_dev=Average_deviation  

  for index, row in DataSet_for_Generation_322Features.iterrows():
      Element_dict=chemparse.parse_formula(row["formula"]) 
      Elem = row["formula"]
      comp = Composition(Elem) 

      electronegativ_List=[]
      electronegativ_List_Based_on_fraction=[]                    
      elem_electronegativ_Subscript_List=[] 
      pettifor_number_List=[]  
      pettifor_number_List_Based_on_fraction=[]                      
      elem_pettifor_number_Subscript_List=[] 
      VanderwaalsR_List=[]  
      period_number_List=[] 
      group_number_List=[]     
      heat_formation_List=[]  
      heat_formation_List_Based_on_fraction=[]                     
      elem_heat_formation_Subscript_List=[] 
      unpaired_electron_List=[] 
      unpaired_electron_List_Based_on_fraction=[]                 
      elem_unpaired_electron_Subscript_List=[]
      Number_Elec_Valence_List=[]  
      Number_Elec_Valence_List_Based_on_fraction=[]                   
      elem_Number_Elec_Valence_Subscript_List=[]  
      MagMoment_List=[]  
      MagMoment_List_Based_on_fraction=[]                 
      elem_MagMoment_Subscript_List=[]  
      dipole_polarizability_List=[]  
      dipole_polarizability_List_Based_on_fraction=[]                      
      elem_dipole_polarizability_Subscript_List=[]  
      First_Ionis_Energy_List=[]  
      First_Ionis_Energy_List_Based_on_fraction=[]                     
      elem_First_Ionis_Energy_Subscript_List=[]  
      ElecAffinity_List=[]  
      ElecAffinity_List_Based_on_fraction=[]                        
      elem_ElecAffinity_Subscript_List=[]  
      Thermal_conduct_List=[]  
      Thermal_conduct_List_Based_on_fraction=[]                    
      elem_Thermal_conduct_Subscript_List=[]  
      Electric_Conduct_List=[]  
      Electric_Conduct_List_Based_on_fraction=[]                     
      elem_Electric_Conduct_Subscript_List=[]  
      Specific_heat_List=[]  
      Specific_heat_List_Based_on_fraction=[]               
      elem_Specific_heat_Subscript_List=[] 
      Ionic_Radius_List=[]  

      for k, v in Element_dict.items(): # k=key and v= value
          elem_str=element(str(k))

          elem_electronegativ=elem_str.electronegativity('pauling')
          if elem_electronegativ==None:
              elem_electronegativ=0.1
          if k == 'Kr':
              elem_electronegativ=3
          electronegativ_List.append(elem_electronegativ)       
          fraction_Elements = comp.get_atomic_fraction(k)
          electronegativ_Based_on_fraction=elem_electronegativ*fraction_Elements 
          electronegativ_List_Based_on_fraction.append(electronegativ_Based_on_fraction)
          elem_electronegativ_Subscript=(elem_electronegativ*v)
          elem_electronegativ_Subscript_List.append(elem_electronegativ_Subscript)

          elem_pettifor_number=elem_str.pettifor_number 
          if elem_pettifor_number==None:
              elem_pettifor_number=120
          pettifor_number_List.append(elem_pettifor_number)       
          pettifor_number_Based_on_fraction=elem_pettifor_number*fraction_Elements 
          pettifor_number_List_Based_on_fraction.append(pettifor_number_Based_on_fraction)
          elem_pettifor_number_Subscript=(elem_pettifor_number*v)
          elem_pettifor_number_Subscript_List.append(elem_pettifor_number_Subscript)

          elem_VanderwaalsR=elem_str.vdw_radius 
          if elem_VanderwaalsR==None:
              elem_VanderwaalsR=120     
          VanderwaalsR_List.append(elem_VanderwaalsR)  

          elem_period_number=elem_str.period
          if elem_period_number==None:
              elem_period_number=3
          period_number_List.append(elem_period_number) 

          elem_group_number=elem_str.group_id
          if elem_group_number==None:
              elem_group_number=4
          group_number_List.append(elem_group_number) 

          elem_heat_formation=elem_str.heat_of_formation 
          if elem_heat_formation==None:
              elem_heat_formation=170    
          heat_formation_List.append(elem_heat_formation)         
          heat_formation_Based_on_fraction=elem_heat_formation*fraction_Elements 
          heat_formation_List_Based_on_fraction.append(heat_formation_Based_on_fraction)
          elem_heat_formation_Subscript=(elem_heat_formation*v)
          elem_heat_formation_Subscript_List.append(elem_heat_formation_Subscript)     

          elem_unpaired_electron=elem_str.ec.unpaired_electrons() 
          if elem_unpaired_electron==None:
              elem_unpaired_electron=1
          unpaired_electron_List.append(elem_unpaired_electron)        
          unpaired_electron_Based_on_fraction=elem_unpaired_electron*fraction_Elements 
          unpaired_electron_List_Based_on_fraction.append(unpaired_electron_Based_on_fraction)
          elem_unpaired_electron_Subscript=(elem_unpaired_electron*v)
          elem_unpaired_electron_Subscript_List.append(elem_unpaired_electron_Subscript)

          elem_Number_Elec_Valence=elem_str.nvalence()
          if elem_Number_Elec_Valence==None:
              elem_Number_Elec_Valence=2
          Number_Elec_Valence_List.append(elem_Number_Elec_Valence)           
          Number_Elec_Valence_Based_on_fraction=elem_Number_Elec_Valence*fraction_Elements 
          Number_Elec_Valence_List_Based_on_fraction.append(Number_Elec_Valence_Based_on_fraction)
          elem_Number_Elec_Valence_Subscript=(elem_Number_Elec_Valence*v)
          elem_Number_Elec_Valence_Subscript_List.append(elem_Number_Elec_Valence_Subscript)

          elem_MagMoment=elem_str.ec.spin_only_magnetic_moment()
          if elem_MagMoment==None:
              elem_MagMoment=0.5
          MagMoment_List.append(elem_MagMoment)       
          MagMoment_Based_on_fraction=elem_MagMoment*fraction_Elements 
          MagMoment_List_Based_on_fraction.append(MagMoment_Based_on_fraction)
          elem_MagMoment_Subscript=(elem_MagMoment*v)
          elem_MagMoment_Subscript_List.append(elem_MagMoment_Subscript)

          elem_dipole_polarizability=elem_str.dipole_polarizability
          if elem_dipole_polarizability==None:
              elem_dipole_polarizability=75
          dipole_polarizability_List.append(elem_dipole_polarizability)       
          dipole_polarizability_Based_on_fraction=elem_dipole_polarizability*fraction_Elements 
          dipole_polarizability_List_Based_on_fraction.append(dipole_polarizability_Based_on_fraction)
          elem_dipole_polarizability_Subscript=(elem_dipole_polarizability*v)
          elem_dipole_polarizability_Subscript_List.append(elem_dipole_polarizability_Subscript)

          Atomic_number_element=elem_str.atomic_number
          elem_First_Ionis_Energy =fetch_ionization_energies(degree=1)['IE1'][Atomic_number_element]
          if elem_First_Ionis_Energy==None:
              elem_First_Ionis_Energy=1
          First_Ionis_Energy_List.append(elem_First_Ionis_Energy)       
          First_Ionis_Energy_Based_on_fraction=elem_First_Ionis_Energy*fraction_Elements 
          First_Ionis_Energy_List_Based_on_fraction.append(First_Ionis_Energy_Based_on_fraction)
          elem_First_Ionis_Energy_Subscript=(elem_First_Ionis_Energy*v)
          elem_First_Ionis_Energy_Subscript_List.append(elem_First_Ionis_Energy_Subscript)

          ElecAffinity=ElecAffinity_Wiki[ElecAffinity_Wiki['element']==str(k)]['Electron_affinity_Wiki_Kj_mol'].values[0]
          ElecAffinity=float(ElecAffinity)
          ElecAffinity_List.append(ElecAffinity)       
          ElecAffinity_Based_on_fraction=ElecAffinity*fraction_Elements 
          ElecAffinity_List_Based_on_fraction.append(ElecAffinity_Based_on_fraction)
          elem_ElecAffinity_Subscript=(ElecAffinity*v)
          elem_ElecAffinity_Subscript_List.append(elem_ElecAffinity_Subscript)
    
          Thermal_Conductiv=Thermal_Conductivity[Thermal_Conductivity['element']==str(k)]['Thermal_Conductivity_W_mk'].values[0]
          Thermal_Conductiv=float(Thermal_Conductiv)
          Thermal_conduct_List.append(Thermal_Conductiv)       
          Thermal_conduct_Based_on_fraction=Thermal_Conductiv*fraction_Elements 
          Thermal_conduct_List_Based_on_fraction.append(Thermal_conduct_Based_on_fraction)
          elem_Thermal_conduct_Subscript=(Thermal_Conductiv*v)
          elem_Thermal_conduct_Subscript_List.append(elem_Thermal_conduct_Subscript)

          Electrical_Conductiv=Electrical_Conductivity[Electrical_Conductivity['element']==str(k)]['Elec_Conductivity_MS_m'].values[0]
          Electrical_Conductiv=float(Electrical_Conductiv)
          if Electrical_Conductiv==None:
              Electrical_Conductiv=1
          Electric_Conduct_List.append(Electrical_Conductiv)       
          Electric_Conduct_Based_on_fraction=Electrical_Conductiv*fraction_Elements 
          Electric_Conduct_List_Based_on_fraction.append(Electric_Conduct_Based_on_fraction)
          elem_Electric_Conduct_Subscript=(Electrical_Conductiv*v)
          elem_Electric_Conduct_Subscript_List.append(elem_Electric_Conduct_Subscript)

          Specific_heat_1=Specific_heat[Specific_heat['element']==str(k)]['Specific_heat_J_KgK'].values[0]
          Specific_heat_1=float(Specific_heat_1)
          if Specific_heat_1==None:
              Specific_heat_1=200
          Specific_heat_List.append(Specific_heat_1)       
          Specific_heat_Based_on_fraction=Specific_heat_1*fraction_Elements 
          Specific_heat_List_Based_on_fraction.append(Specific_heat_Based_on_fraction)
          elem_Specific_heat_Subscript=(Specific_heat_1*v)
          elem_Specific_heat_Subscript_List.append(elem_Specific_heat_Subscript)

          Ionic_Radius=df_Ionic_Radius_Element_periodic_tabel[df_Ionic_Radius_Element_periodic_tabel['element']==str(k)]['Ionic_Radius_Based_Angestrom'].values[0]
          Ionic_Radius=float(Ionic_Radius)
          if Ionic_Radius==None:
              Ionic_Radius=1
          Ionic_Radius_List.append(Ionic_Radius)       

      total_electronegativ_Based_fraction_mean.append(np.mean(electronegativ_List_Based_on_fraction))
      total_electronegativ_Based_fraction_median.append(np.median(electronegativ_List_Based_on_fraction))
      total_electronegativ_Based_fraction_variance.append(np.var(electronegativ_List_Based_on_fraction))
      total_electronegativ_Based_fraction_max.append(np.max(electronegativ_List_Based_on_fraction))
      total_electronegativ_Based_fraction_min.append(np.min(electronegativ_List_Based_on_fraction))
      total_electronegativ_Based_fraction_range.append(np.ptp(electronegativ_List_Based_on_fraction))
      total_electronegativ_Based_fraction_std.append(np.std(electronegativ_List_Based_on_fraction))
      # Average_deviation calculate 
      mean_value = np.mean(electronegativ_List_Based_on_fraction)
      res=[]
      for ele in electronegativ_List_Based_on_fraction:
        res.append(abs(ele - mean_value))
      total_electronegativ_Based_fraction_Ave_dev.append(np.mean(res))
      total_electronegativ_Based_Elemental_mean.append(np.mean(electronegativ_List))
      total_electronegativ_Based_Elemental_median.append(np.median(electronegativ_List))
      total_electronegativ_Based_Elemental_variance.append(np.var(electronegativ_List))
      total_electronegativ_Based_Elemental_max.append(np.max(electronegativ_List))
      total_electronegativ_Based_Elemental_min.append(np.min(electronegativ_List))
      total_electronegativ_Based_Elemental_range.append(np.ptp(electronegativ_List))
      total_electronegativ_Based_Elemental_std.append(np.std(electronegativ_List)) 
      # Average_deviation calculate
      mean_value = np.mean(electronegativ_List)
      res=[]
      for ele in electronegativ_List:
        res.append(abs(ele - mean_value))
      total_electronegativ_Based_Elemental_Ave_dev.append(np.mean(res))
      total_electronegativ_Based_Subscript_mean.append(np.mean(elem_electronegativ_Subscript_List))
      total_electronegativ_Based_Subscript_median.append(np.median(elem_electronegativ_Subscript_List))
      total_electronegativ_Based_Subscript_variance.append(np.var(elem_electronegativ_Subscript_List))
      total_electronegativ_Based_Subscript_max.append(np.max(elem_electronegativ_Subscript_List))
      total_electronegativ_Based_Subscript_min.append(np.min(elem_electronegativ_Subscript_List))
      total_electronegativ_Based_Subscript_range.append(np.ptp(elem_electronegativ_Subscript_List))
      total_electronegativ_Based_Subscript_std.append(np.std(elem_electronegativ_Subscript_List)) 
      # Average_deviation calculate
      mean_value = np.mean(elem_electronegativ_Subscript_List)
      res=[]
      for ele in elem_electronegativ_Subscript_List:
        res.append(abs(ele - mean_value))
      total_electronegativ_Based_Subscript_Ave_dev.append(np.mean(res))

      total_pettifor_number_Based_fraction_mean.append(np.mean(pettifor_number_List_Based_on_fraction))
      total_pettifor_number_Based_fraction_median.append(np.median(pettifor_number_List_Based_on_fraction))
      total_pettifor_number_Based_fraction_variance.append(np.var(pettifor_number_List_Based_on_fraction))
      total_pettifor_number_Based_fraction_max.append(np.max(pettifor_number_List_Based_on_fraction))
      total_pettifor_number_Based_fraction_min.append(np.min(pettifor_number_List_Based_on_fraction))
      total_pettifor_number_Based_fraction_range.append(np.ptp(pettifor_number_List_Based_on_fraction))
      total_pettifor_number_Based_fraction_std.append(np.std(pettifor_number_List_Based_on_fraction)) 
      # Average_deviation calculate
      mean_value = np.mean(pettifor_number_List_Based_on_fraction)
      res=[]
      for ele in pettifor_number_List_Based_on_fraction:
        res.append(abs(ele - mean_value))
      total_pettifor_number_Based_fraction_Ave_dev.append(np.mean(res))
      total_pettifor_number_Based_Elemental_mean.append(np.mean(pettifor_number_List))
      total_pettifor_number_Based_Elemental_median.append(np.median(pettifor_number_List))
      total_pettifor_number_Based_Elemental_variance.append(np.var(pettifor_number_List))
      total_pettifor_number_Based_Elemental_max.append(np.max(pettifor_number_List))
      total_pettifor_number_Based_Elemental_min.append(np.min(pettifor_number_List))
      total_pettifor_number_Based_Elemental_range.append(np.ptp(pettifor_number_List))
      total_pettifor_number_Based_Elemental_std.append(np.std(pettifor_number_List)) 
      # Average_deviation calculate
      mean_value = np.mean(pettifor_number_List)
      res=[]
      for ele in pettifor_number_List:
        res.append(abs(ele - mean_value))
      total_pettifor_number_Based_Elemental_Ave_dev.append(np.mean(res))
      total_pettifor_number_Based_Subscript_mean.append(np.mean(elem_pettifor_number_Subscript_List))
      total_pettifor_number_Based_Subscript_median.append(np.median(elem_pettifor_number_Subscript_List))
      total_pettifor_number_Based_Subscript_variance.append(np.var(elem_pettifor_number_Subscript_List))
      total_pettifor_number_Based_Subscript_max.append(np.max(elem_pettifor_number_Subscript_List))
      total_pettifor_number_Based_Subscript_min.append(np.min(elem_pettifor_number_Subscript_List))
      total_pettifor_number_Based_Subscript_range.append(np.ptp(elem_pettifor_number_Subscript_List))
      total_pettifor_number_Based_Subscript_std.append(np.std(elem_pettifor_number_Subscript_List)) 
      # Average_deviation calculate
      mean_value = np.mean(elem_pettifor_number_Subscript_List)
      res=[]
      for ele in elem_pettifor_number_Subscript_List:
        res.append(abs(ele - mean_value))
      total_pettifor_number_Based_Subscript_Ave_dev.append(np.mean(res))

      total_VanderwaalsR_Based_Elemental_mean.append(np.mean(VanderwaalsR_List))
      total_VanderwaalsR_Based_Elemental_median.append(np.median(VanderwaalsR_List))
      total_VanderwaalsR_Based_Elemental_variance.append(np.var(VanderwaalsR_List))
      total_VanderwaalsR_Based_Elemental_max.append(np.max(VanderwaalsR_List))
      total_VanderwaalsR_Based_Elemental_min.append(np.min(VanderwaalsR_List))
      total_VanderwaalsR_Based_Elemental_range.append(np.ptp(VanderwaalsR_List))
      total_VanderwaalsR_Based_Elemental_std.append(np.std(VanderwaalsR_List)) 
      # Average_deviation calculate
      mean_value = np.mean(VanderwaalsR_List)
      res=[]
      for ele in VanderwaalsR_List:
        res.append(abs(ele - mean_value))
      total_VanderwaalsR_Based_Elemental_Ave_dev.append(np.mean(res))

      total_period_number_Based_Elemental_mean.append(np.mean(period_number_List))
      total_period_number_Based_Elemental_median.append(np.median(period_number_List))
      total_period_number_Based_Elemental_variance.append(np.var(period_number_List))
      total_period_number_Based_Elemental_max.append(np.max(period_number_List))
      total_period_number_Based_Elemental_min.append(np.min(period_number_List))
      total_period_number_Based_Elemental_range.append(np.ptp(period_number_List))
      total_period_number_Based_Elemental_std.append(np.std(period_number_List)) 
      # Average_deviation calculate
      mean_value = np.mean(period_number_List)
      res=[]
      for ele in period_number_List:
        res.append(abs(ele - mean_value))
      total_period_number_Based_Elemental_Ave_dev.append(np.mean(res))

      total_group_number_Based_Elemental_mean.append(np.mean(group_number_List))
      total_group_number_Based_Elemental_median.append(np.median(group_number_List))
      total_group_number_Based_Elemental_variance.append(np.var(group_number_List))
      total_group_number_Based_Elemental_max.append(np.max(group_number_List))
      total_group_number_Based_Elemental_min.append(np.min(group_number_List))
      total_group_number_Based_Elemental_range.append(np.ptp(group_number_List))
      total_group_number_Based_Elemental_std.append(np.std(group_number_List)) 
      # Average_deviation calculate
      mean_value = np.mean(group_number_List)
      res=[]
      for ele in group_number_List:
        res.append(abs(ele - mean_value))
      total_group_number_Based_Elemental_Ave_dev.append(np.mean(res))

      total_heat_formation_Based_fraction_mean.append(np.mean(heat_formation_List_Based_on_fraction))
      total_heat_formation_Based_fraction_median.append(np.median(heat_formation_List_Based_on_fraction))
      total_heat_formation_Based_fraction_variance.append(np.var(heat_formation_List_Based_on_fraction))
      total_heat_formation_Based_fraction_max.append(np.max(heat_formation_List_Based_on_fraction))
      total_heat_formation_Based_fraction_min.append(np.min(heat_formation_List_Based_on_fraction))
      total_heat_formation_Based_fraction_range.append(np.ptp(heat_formation_List_Based_on_fraction))
      total_heat_formation_Based_fraction_std.append(np.std(heat_formation_List_Based_on_fraction)) 
      # Average_deviation calculate
      mean_value = np.mean(heat_formation_List_Based_on_fraction)
      res=[]
      for ele in heat_formation_List_Based_on_fraction:
        res.append(abs(ele - mean_value))
      total_heat_formation_Based_fraction_Ave_dev.append(np.mean(res))
      total_heat_formation_Based_Elemental_mean.append(np.mean(heat_formation_List))
      total_heat_formation_Based_Elemental_median.append(np.median(heat_formation_List))
      total_heat_formation_Based_Elemental_variance.append(np.var(heat_formation_List))
      total_heat_formation_Based_Elemental_max.append(np.max(heat_formation_List))
      total_heat_formation_Based_Elemental_min.append(np.min(heat_formation_List))
      total_heat_formation_Based_Elemental_range.append(np.ptp(heat_formation_List))
      total_heat_formation_Based_Elemental_std.append(np.std(heat_formation_List)) 
      # Average_deviation calculate
      mean_value = np.mean(heat_formation_List)
      res=[]
      for ele in heat_formation_List:
        res.append(abs(ele - mean_value))
      total_heat_formation_Based_Elemental_Ave_dev.append(np.mean(res))
      total_heat_formation_Based_Subscript_mean.append(np.mean(elem_heat_formation_Subscript_List))
      total_heat_formation_Based_Subscript_median.append(np.median(elem_heat_formation_Subscript_List))
      total_heat_formation_Based_Subscript_variance.append(np.var(elem_heat_formation_Subscript_List))
      total_heat_formation_Based_Subscript_max.append(np.max(elem_heat_formation_Subscript_List))
      total_heat_formation_Based_Subscript_min.append(np.min(elem_heat_formation_Subscript_List))
      total_heat_formation_Based_Subscript_range.append(np.ptp(elem_heat_formation_Subscript_List))
      total_heat_formation_Based_Subscript_std.append(np.std(elem_heat_formation_Subscript_List)) 
      # Average_deviation calculate
      mean_value = np.mean(elem_heat_formation_Subscript_List)
      res=[]
      for ele in elem_heat_formation_Subscript_List:
        res.append(abs(ele - mean_value))
      total_heat_formation_Based_Subscript_Ave_dev.append(np.mean(res))

      total_unpaired_electron_Based_fraction_mean.append(np.mean(unpaired_electron_List_Based_on_fraction))
      total_unpaired_electron_Based_fraction_median.append(np.median(unpaired_electron_List_Based_on_fraction))
      total_unpaired_electron_Based_fraction_variance.append(np.var(unpaired_electron_List_Based_on_fraction))
      total_unpaired_electron_Based_fraction_max.append(np.max(unpaired_electron_List_Based_on_fraction))
      total_unpaired_electron_Based_fraction_min.append(np.min(unpaired_electron_List_Based_on_fraction))
      total_unpaired_electron_Based_fraction_range.append(np.ptp(unpaired_electron_List_Based_on_fraction))
      total_unpaired_electron_Based_fraction_std.append(np.std(unpaired_electron_List_Based_on_fraction)) 
      # Average_deviation calculate
      mean_value = np.mean(unpaired_electron_List_Based_on_fraction)
      res=[]
      for ele in unpaired_electron_List_Based_on_fraction:
        res.append(abs(ele - mean_value))
      total_unpaired_electron_Based_fraction_Ave_dev.append(np.mean(res))
      total_unpaired_electron_Based_Elemental_mean.append(np.mean(unpaired_electron_List))
      total_unpaired_electron_Based_Elemental_median.append(np.median(unpaired_electron_List))
      total_unpaired_electron_Based_Elemental_variance.append(np.var(unpaired_electron_List))
      total_unpaired_electron_Based_Elemental_max.append(np.max(unpaired_electron_List))
      total_unpaired_electron_Based_Elemental_min.append(np.min(unpaired_electron_List))
      total_unpaired_electron_Based_Elemental_range.append(np.ptp(unpaired_electron_List)) 
      total_unpaired_electron_Based_Elemental_std.append(np.std(unpaired_electron_List)) 
      # Average_deviation calculate
      mean_value = np.mean(unpaired_electron_List)
      res=[]
      for ele in unpaired_electron_List:
        res.append(abs(ele - mean_value))
      total_unpaired_electron_Based_Elemental_Ave_dev.append(np.mean(res))
      total_unpaired_electron_Based_Subscript_mean.append(np.mean(elem_unpaired_electron_Subscript_List))
      total_unpaired_electron_Based_Subscript_median.append(np.median(elem_unpaired_electron_Subscript_List))
      total_unpaired_electron_Based_Subscript_variance.append(np.var(elem_unpaired_electron_Subscript_List))
      total_unpaired_electron_Based_Subscript_max.append(np.max(elem_unpaired_electron_Subscript_List))
      total_unpaired_electron_Based_Subscript_min.append(np.min(elem_unpaired_electron_Subscript_List))
      total_unpaired_electron_Based_Subscript_range.append(np.ptp(elem_unpaired_electron_Subscript_List))
      total_unpaired_electron_Based_Subscript_std.append(np.std(elem_unpaired_electron_Subscript_List)) 
      # Average_deviation calculate
      mean_value = np.mean(elem_unpaired_electron_Subscript_List)
      res=[]
      for ele in elem_unpaired_electron_Subscript_List:
        res.append(abs(ele - mean_value))
      total_unpaired_electron_Based_Subscript_Ave_dev.append(np.mean(res))

      total_Number_Elec_Valence_Based_fraction_mean.append(np.mean(Number_Elec_Valence_List_Based_on_fraction))
      total_Number_Elec_Valence_Based_fraction_median.append(np.median(Number_Elec_Valence_List_Based_on_fraction))
      total_Number_Elec_Valence_Based_fraction_variance.append(np.var(Number_Elec_Valence_List_Based_on_fraction))
      total_Number_Elec_Valence_Based_fraction_max.append(np.max(Number_Elec_Valence_List_Based_on_fraction))
      total_Number_Elec_Valence_Based_fraction_min.append(np.min(Number_Elec_Valence_List_Based_on_fraction))
      total_Number_Elec_Valence_Based_fraction_range.append(np.ptp(Number_Elec_Valence_List_Based_on_fraction))
      total_Number_Elec_Valence_Based_fraction_std.append(np.std(Number_Elec_Valence_List_Based_on_fraction)) 
      # Average_deviation calculate
      mean_value = np.mean(Number_Elec_Valence_List_Based_on_fraction)
      res=[]
      for ele in Number_Elec_Valence_List_Based_on_fraction:
        res.append(abs(ele - mean_value))
      total_Number_Elec_Valence_Based_fraction_Ave_dev.append(np.mean(res))
      total_Number_Elec_Valence_Based_Elemental_mean.append(np.mean(Number_Elec_Valence_List))
      total_Number_Elec_Valence_Based_Elemental_median.append(np.median(Number_Elec_Valence_List))
      total_Number_Elec_Valence_Based_Elemental_variance.append(np.var(Number_Elec_Valence_List))
      total_Number_Elec_Valence_Based_Elemental_max.append(np.max(Number_Elec_Valence_List))
      total_Number_Elec_Valence_Based_Elemental_min.append(np.min(Number_Elec_Valence_List))
      total_Number_Elec_Valence_Based_Elemental_range.append(np.ptp(Number_Elec_Valence_List))
      total_Number_Elec_Valence_Based_Elemental_std.append(np.std(Number_Elec_Valence_List)) 
      # Average_deviation calculate
      mean_value = np.mean(Number_Elec_Valence_List)
      res=[]
      for ele in Number_Elec_Valence_List:
        res.append(abs(ele - mean_value))
      total_Number_Elec_Valence_Based_Elemental_Ave_dev.append(np.mean(res))
      total_Number_Elec_Valence_Based_Subscript_mean.append(np.mean(elem_Number_Elec_Valence_Subscript_List))
      total_Number_Elec_Valence_Based_Subscript_median.append(np.median(elem_Number_Elec_Valence_Subscript_List))
      total_Number_Elec_Valence_Based_Subscript_variance.append(np.var(elem_Number_Elec_Valence_Subscript_List))
      total_Number_Elec_Valence_Based_Subscript_max.append(np.max(elem_Number_Elec_Valence_Subscript_List))
      total_Number_Elec_Valence_Based_Subscript_min.append(np.min(elem_Number_Elec_Valence_Subscript_List))
      total_Number_Elec_Valence_Based_Subscript_range.append(np.ptp(elem_Number_Elec_Valence_Subscript_List))
      total_Number_Elec_Valence_Based_Subscript_std.append(np.std(elem_Number_Elec_Valence_Subscript_List)) 
      # Average_deviation calculate
      mean_value = np.mean(elem_Number_Elec_Valence_Subscript_List)
      res=[]
      for ele in elem_Number_Elec_Valence_Subscript_List:
        res.append(abs(ele - mean_value))
      total_Number_Elec_Valence_Based_Subscript_Ave_dev.append(np.mean(res))

      total_MagMoment_Based_fraction_mean.append(np.mean(MagMoment_List_Based_on_fraction))
      total_MagMoment_Based_fraction_median.append(np.median(MagMoment_List_Based_on_fraction))
      total_MagMoment_Based_fraction_variance.append(np.var(MagMoment_List_Based_on_fraction))
      total_MagMoment_Based_fraction_max.append(np.max(MagMoment_List_Based_on_fraction))
      total_MagMoment_Based_fraction_min.append(np.min(MagMoment_List_Based_on_fraction))
      total_MagMoment_Based_fraction_range.append(np.ptp(MagMoment_List_Based_on_fraction))
      total_MagMoment_Based_fraction_std.append(np.std(MagMoment_List_Based_on_fraction)) 
      # Average_deviation calculate
      mean_value = np.mean(MagMoment_List_Based_on_fraction)
      res=[]
      for ele in MagMoment_List_Based_on_fraction:
        res.append(abs(ele - mean_value))
      total_MagMoment_Based_fraction_Ave_dev.append(np.mean(res))
      total_MagMoment_Based_Elemental_mean.append(np.mean(MagMoment_List))
      total_MagMoment_Based_Elemental_median.append(np.median(MagMoment_List))
      total_MagMoment_Based_Elemental_variance.append(np.var(MagMoment_List))
      total_MagMoment_Based_Elemental_max.append(np.max(MagMoment_List))
      total_MagMoment_Based_Elemental_min.append(np.min(MagMoment_List))
      total_MagMoment_Based_Elemental_range.append(np.ptp(MagMoment_List))
      total_MagMoment_Based_Elemental_std.append(np.std(MagMoment_List)) 
      # Average_deviation calculate
      mean_value = np.mean(MagMoment_List)
      res=[]
      for ele in MagMoment_List:
        res.append(abs(ele - mean_value))
      total_MagMoment_Based_Elemental_Ave_dev.append(np.mean(res))
      total_MagMoment_Based_Subscript_mean.append(np.mean(elem_MagMoment_Subscript_List))
      total_MagMoment_Based_Subscript_median.append(np.median(elem_MagMoment_Subscript_List))
      total_MagMoment_Based_Subscript_variance.append(np.var(elem_MagMoment_Subscript_List))
      total_MagMoment_Based_Subscript_max.append(np.max(elem_MagMoment_Subscript_List))
      total_MagMoment_Based_Subscript_min.append(np.min(elem_MagMoment_Subscript_List))
      total_MagMoment_Based_Subscript_range.append(np.ptp(elem_MagMoment_Subscript_List))
      total_MagMoment_Based_Subscript_std.append(np.std(elem_MagMoment_Subscript_List)) 
      # Average_deviation calculate
      mean_value = np.mean(elem_MagMoment_Subscript_List)
      res=[]
      for ele in elem_MagMoment_Subscript_List:
        res.append(abs(ele - mean_value))
      total_MagMoment_Based_Subscript_Ave_dev.append(np.mean(res))

      total_dipole_polarizability_Based_fraction_mean.append(np.mean(dipole_polarizability_List_Based_on_fraction))
      total_dipole_polarizability_Based_fraction_median.append(np.median(dipole_polarizability_List_Based_on_fraction))
      total_dipole_polarizability_Based_fraction_variance.append(np.var(dipole_polarizability_List_Based_on_fraction))
      total_dipole_polarizability_Based_fraction_max.append(np.max(dipole_polarizability_List_Based_on_fraction))
      total_dipole_polarizability_Based_fraction_min.append(np.min(dipole_polarizability_List_Based_on_fraction))
      total_dipole_polarizability_Based_fraction_range.append(np.ptp(dipole_polarizability_List_Based_on_fraction))
      total_dipole_polarizability_Based_fraction_std.append(np.std(dipole_polarizability_List_Based_on_fraction)) 
      # Average_deviation calculate
      mean_value = np.mean(dipole_polarizability_List_Based_on_fraction)
      res=[]
      for ele in dipole_polarizability_List_Based_on_fraction:
        res.append(abs(ele - mean_value))
      total_dipole_polarizability_Based_fraction_Ave_dev.append(np.mean(res))
      total_dipole_polarizability_Based_Elemental_mean.append(np.mean(dipole_polarizability_List))
      total_dipole_polarizability_Based_Elemental_median.append(np.median(dipole_polarizability_List))
      total_dipole_polarizability_Based_Elemental_variance.append(np.var(dipole_polarizability_List))
      total_dipole_polarizability_Based_Elemental_max.append(np.max(dipole_polarizability_List))
      total_dipole_polarizability_Based_Elemental_min.append(np.min(dipole_polarizability_List))
      total_dipole_polarizability_Based_Elemental_range.append(np.ptp(dipole_polarizability_List))
      total_dipole_polarizability_Based_Elemental_std.append(np.std(dipole_polarizability_List)) 
      # Average_deviation calculate
      mean_value = np.mean(dipole_polarizability_List)
      res=[]
      for ele in dipole_polarizability_List:
        res.append(abs(ele - mean_value))
      total_dipole_polarizability_Based_Elemental_Ave_dev.append(np.mean(res))
      total_dipole_polarizability_Based_Subscript_mean.append(np.mean(elem_dipole_polarizability_Subscript_List))
      total_dipole_polarizability_Based_Subscript_median.append(np.median(elem_dipole_polarizability_Subscript_List))
      total_dipole_polarizability_Based_Subscript_variance.append(np.var(elem_dipole_polarizability_Subscript_List))
      total_dipole_polarizability_Based_Subscript_max.append(np.max(elem_dipole_polarizability_Subscript_List))
      total_dipole_polarizability_Based_Subscript_min.append(np.min(elem_dipole_polarizability_Subscript_List))
      total_dipole_polarizability_Based_Subscript_range.append(np.ptp(elem_dipole_polarizability_Subscript_List))
      total_dipole_polarizability_Based_Subscript_std.append(np.std(elem_dipole_polarizability_Subscript_List)) 
      # Average_deviation calculate
      mean_value = np.mean(elem_dipole_polarizability_Subscript_List)
      res=[]
      for ele in elem_dipole_polarizability_Subscript_List:
        res.append(abs(ele - mean_value))
      total_dipole_polarizability_Based_Subscript_Ave_dev.append(np.mean(res))

      total_First_Ionis_Energy_Based_fraction_mean.append(np.mean(First_Ionis_Energy_List_Based_on_fraction))
      total_First_Ionis_Energy_Based_fraction_median.append(np.median(First_Ionis_Energy_List_Based_on_fraction))
      total_First_Ionis_Energy_Based_fraction_variance.append(np.var(First_Ionis_Energy_List_Based_on_fraction))
      total_First_Ionis_Energy_Based_fraction_max.append(np.max(First_Ionis_Energy_List_Based_on_fraction))
      total_First_Ionis_Energy_Based_fraction_min.append(np.min(First_Ionis_Energy_List_Based_on_fraction))
      total_First_Ionis_Energy_Based_fraction_range.append(np.ptp(First_Ionis_Energy_List_Based_on_fraction))
      total_First_Ionis_Energy_Based_fraction_std.append(np.std(First_Ionis_Energy_List_Based_on_fraction)) 
      # Average_deviation calculate
      mean_value = np.mean(First_Ionis_Energy_List_Based_on_fraction)
      res=[]
      for ele in First_Ionis_Energy_List_Based_on_fraction:
        res.append(abs(ele - mean_value))
      total_First_Ionis_Energy_Based_fraction_Ave_dev.append(np.mean(res))
      total_First_Ionis_Energy_Based_Elemental_mean.append(np.mean(First_Ionis_Energy_List))
      total_First_Ionis_Energy_Based_Elemental_median.append(np.median(First_Ionis_Energy_List))
      total_First_Ionis_Energy_Based_Elemental_variance.append(np.var(First_Ionis_Energy_List))
      total_First_Ionis_Energy_Based_Elemental_max.append(np.max(First_Ionis_Energy_List))
      total_First_Ionis_Energy_Based_Elemental_min.append(np.min(First_Ionis_Energy_List))
      total_First_Ionis_Energy_Based_Elemental_range.append(np.ptp(First_Ionis_Energy_List))
      total_First_Ionis_Energy_Based_Elemental_std.append(np.std(First_Ionis_Energy_List)) 
      # Average_deviation calculate
      mean_value = np.mean(First_Ionis_Energy_List)
      res=[]
      for ele in First_Ionis_Energy_List:
        res.append(abs(ele - mean_value))
      total_First_Ionis_Energy_Based_Elemental_Ave_dev.append(np.mean(res))
      total_First_Ionis_Energy_Based_Subscript_mean.append(np.mean(elem_First_Ionis_Energy_Subscript_List))
      total_First_Ionis_Energy_Based_Subscript_median.append(np.median(elem_First_Ionis_Energy_Subscript_List))
      total_First_Ionis_Energy_Based_Subscript_variance.append(np.var(elem_First_Ionis_Energy_Subscript_List))
      total_First_Ionis_Energy_Based_Subscript_max.append(np.max(elem_First_Ionis_Energy_Subscript_List))
      total_First_Ionis_Energy_Based_Subscript_min.append(np.min(elem_First_Ionis_Energy_Subscript_List))
      total_First_Ionis_Energy_Based_Subscript_range.append(np.ptp(elem_First_Ionis_Energy_Subscript_List))
      total_First_Ionis_Energy_Based_Subscript_std.append(np.std(elem_First_Ionis_Energy_Subscript_List)) 
      # Average_deviation calculate
      mean_value = np.mean(elem_First_Ionis_Energy_Subscript_List)
      res=[]
      for ele in elem_First_Ionis_Energy_Subscript_List:
        res.append(abs(ele - mean_value))
      total_First_Ionis_Energy_Based_Subscript_Ave_dev.append(np.mean(res))

      total_ElecAffinity_Based_fraction_mean.append(np.mean(ElecAffinity_List_Based_on_fraction))
      total_ElecAffinity_Based_fraction_median.append(np.median(ElecAffinity_List_Based_on_fraction))
      total_ElecAffinity_Based_fraction_variance.append(np.var(ElecAffinity_List_Based_on_fraction))
      total_ElecAffinity_Based_fraction_max.append(np.max(ElecAffinity_List_Based_on_fraction))
      total_ElecAffinity_Based_fraction_min.append(np.min(ElecAffinity_List_Based_on_fraction))
      total_ElecAffinity_Based_fraction_range.append(np.ptp(ElecAffinity_List_Based_on_fraction))
      total_ElecAffinity_Based_fraction_std.append(np.std(ElecAffinity_List_Based_on_fraction)) 
      # Average_deviation calculate
      mean_value = np.mean(ElecAffinity_List_Based_on_fraction)
      res=[]
      for ele in ElecAffinity_List_Based_on_fraction:
        res.append(abs(ele - mean_value))
      total_ElecAffinity_Based_fraction_Ave_dev.append(np.mean(res))
      total_ElecAffinity_Based_Elemental_mean.append(np.mean(ElecAffinity_List))
      total_ElecAffinity_Based_Elemental_median.append(np.median(ElecAffinity_List))
      total_ElecAffinity_Based_Elemental_variance.append(np.var(ElecAffinity_List))
      total_ElecAffinity_Based_Elemental_max.append(np.max(ElecAffinity_List))
      total_ElecAffinity_Based_Elemental_min.append(np.min(ElecAffinity_List))
      total_ElecAffinity_Based_Elemental_range.append(np.ptp(ElecAffinity_List))
      total_ElecAffinity_Based_Elemental_std.append(np.std(ElecAffinity_List)) 
      # Average_deviation calculate
      mean_value = np.mean(ElecAffinity_List)
      res=[]
      for ele in ElecAffinity_List:
        res.append(abs(ele - mean_value))
      total_ElecAffinity_Based_Elemental_Ave_dev.append(np.mean(res))
      total_ElecAffinity_Based_Subscript_mean.append(np.mean(elem_ElecAffinity_Subscript_List))
      total_ElecAffinity_Based_Subscript_median.append(np.median(elem_ElecAffinity_Subscript_List))
      total_ElecAffinity_Based_Subscript_variance.append(np.var(elem_ElecAffinity_Subscript_List))
      total_ElecAffinity_Based_Subscript_max.append(np.max(elem_ElecAffinity_Subscript_List))
      total_ElecAffinity_Based_Subscript_min.append(np.min(elem_ElecAffinity_Subscript_List))
      total_ElecAffinity_Based_Subscript_range.append(np.ptp(elem_ElecAffinity_Subscript_List))
      total_ElecAffinity_Based_Subscript_std.append(np.std(elem_ElecAffinity_Subscript_List)) 
      # Average_deviation calculate
      mean_value = np.mean(elem_ElecAffinity_Subscript_List)
      res=[]
      for ele in elem_ElecAffinity_Subscript_List:
        res.append(abs(ele - mean_value))
      total_ElecAffinity_Based_Subscript_Ave_dev.append(np.mean(res))

      total_Thermal_conduct_Based_fraction_mean.append(np.mean(Thermal_conduct_List_Based_on_fraction))
      total_Thermal_conduct_Based_fraction_median.append(np.median(Thermal_conduct_List_Based_on_fraction))
      total_Thermal_conduct_Based_fraction_variance.append(np.var(Thermal_conduct_List_Based_on_fraction))
      total_Thermal_conduct_Based_fraction_max.append(np.max(Thermal_conduct_List_Based_on_fraction))
      total_Thermal_conduct_Based_fraction_min.append(np.min(Thermal_conduct_List_Based_on_fraction))
      total_Thermal_conduct_Based_fraction_range.append(np.ptp(Thermal_conduct_List_Based_on_fraction))
      total_Thermal_conduct_Based_fraction_std.append(np.std(Thermal_conduct_List_Based_on_fraction)) 
      # Average_deviation calculate
      mean_value = np.mean(Thermal_conduct_List_Based_on_fraction)
      res=[]
      for ele in Thermal_conduct_List_Based_on_fraction:
        res.append(abs(ele - mean_value))
      total_Thermal_conduct_Based_fraction_Ave_dev.append(np.mean(res))
      total_Thermal_conduct_Based_Elemental_mean.append(np.mean(Thermal_conduct_List))
      total_Thermal_conduct_Based_Elemental_median.append(np.median(Thermal_conduct_List))
      total_Thermal_conduct_Based_Elemental_variance.append(np.var(Thermal_conduct_List))
      total_Thermal_conduct_Based_Elemental_max.append(np.max(Thermal_conduct_List))
      total_Thermal_conduct_Based_Elemental_min.append(np.min(Thermal_conduct_List))
      total_Thermal_conduct_Based_Elemental_range.append(np.ptp(Thermal_conduct_List))
      total_Thermal_conduct_Based_Elemental_std.append(np.std(Thermal_conduct_List)) 
      # Average_deviation calculate
      mean_value = np.mean(Thermal_conduct_List)
      res=[]
      for ele in Thermal_conduct_List:
        res.append(abs(ele - mean_value))
      total_Thermal_conduct_Based_Elemental_Ave_dev.append(np.mean(res))
      total_Thermal_conduct_Based_Subscript_mean.append(np.mean(elem_Thermal_conduct_Subscript_List))
      total_Thermal_conduct_Based_Subscript_median.append(np.median(elem_Thermal_conduct_Subscript_List))
      total_Thermal_conduct_Based_Subscript_variance.append(np.var(elem_Thermal_conduct_Subscript_List))
      total_Thermal_conduct_Based_Subscript_max.append(np.max(elem_Thermal_conduct_Subscript_List))
      total_Thermal_conduct_Based_Subscript_min.append(np.min(elem_Thermal_conduct_Subscript_List))
      total_Thermal_conduct_Based_Subscript_range.append(np.ptp(elem_Thermal_conduct_Subscript_List))
      total_Thermal_conduct_Based_Subscript_std.append(np.std(elem_Thermal_conduct_Subscript_List)) 
      # Average_deviation calculate
      mean_value = np.mean(elem_Thermal_conduct_Subscript_List)
      res=[]
      for ele in elem_Thermal_conduct_Subscript_List:
        res.append(abs(ele - mean_value))
      total_Thermal_conduct_Based_Subscript_Ave_dev.append(np.mean(res))

      total_Electric_Conduct_Based_fraction_mean.append(np.mean(Electric_Conduct_List_Based_on_fraction))
      total_Electric_Conduct_Based_fraction_median.append(np.median(Electric_Conduct_List_Based_on_fraction))
      total_Electric_Conduct_Based_fraction_variance.append(np.var(Electric_Conduct_List_Based_on_fraction))
      total_Electric_Conduct_Based_fraction_max.append(np.max(Electric_Conduct_List_Based_on_fraction))
      total_Electric_Conduct_Based_fraction_min.append(np.min(Electric_Conduct_List_Based_on_fraction))
      total_Electric_Conduct_Based_fraction_range.append(np.ptp(Electric_Conduct_List_Based_on_fraction))
      total_Electric_Conduct_Based_fraction_std.append(np.std(Electric_Conduct_List_Based_on_fraction)) 
      # Average_deviation calculate
      mean_value = np.mean(Electric_Conduct_List_Based_on_fraction)
      res=[]
      for ele in Electric_Conduct_List_Based_on_fraction:
        res.append(abs(ele - mean_value))
      total_Electric_Conduct_Based_fraction_Ave_dev.append(np.mean(res))
      total_Electric_Conduct_Based_Elemental_mean.append(np.mean(Electric_Conduct_List))
      total_Electric_Conduct_Based_Elemental_median.append(np.median(Electric_Conduct_List))
      total_Electric_Conduct_Based_Elemental_variance.append(np.var(Electric_Conduct_List))
      total_Electric_Conduct_Based_Elemental_max.append(np.max(Electric_Conduct_List))
      total_Electric_Conduct_Based_Elemental_min.append(np.min(Electric_Conduct_List))
      total_Electric_Conduct_Based_Elemental_range.append(np.ptp(Electric_Conduct_List))
      total_Electric_Conduct_Based_Elemental_std.append(np.std(Electric_Conduct_List)) 
      # Average_deviation calculate
      mean_value = np.mean(Electric_Conduct_List)
      res=[]
      for ele in Electric_Conduct_List:
        res.append(abs(ele - mean_value))
      total_Electric_Conduct_Based_Elemental_Ave_dev.append(np.mean(res))
      total_Electric_Conduct_Based_Subscript_mean.append(np.mean(elem_Electric_Conduct_Subscript_List))
      total_Electric_Conduct_Based_Subscript_median.append(np.median(elem_Electric_Conduct_Subscript_List))
      total_Electric_Conduct_Based_Subscript_variance.append(np.var(elem_Electric_Conduct_Subscript_List))
      total_Electric_Conduct_Based_Subscript_max.append(np.max(elem_Electric_Conduct_Subscript_List))
      total_Electric_Conduct_Based_Subscript_min.append(np.min(elem_Electric_Conduct_Subscript_List))
      total_Electric_Conduct_Based_Subscript_range.append(np.ptp(elem_Electric_Conduct_Subscript_List))
      total_Electric_Conduct_Based_Subscript_std.append(np.std(elem_Electric_Conduct_Subscript_List)) 
      # Average_deviation calculate
      mean_value = np.mean(elem_Electric_Conduct_Subscript_List)
      res=[]
      for ele in elem_Electric_Conduct_Subscript_List:
        res.append(abs(ele - mean_value))
      total_Electric_Conduct_Based_Subscript_Ave_dev.append(np.mean(res))

      total_Specific_heat_Based_fraction_mean.append(np.mean(Specific_heat_List_Based_on_fraction))
      total_Specific_heat_Based_fraction_median.append(np.median(Specific_heat_List_Based_on_fraction))
      total_Specific_heat_Based_fraction_variance.append(np.var(Specific_heat_List_Based_on_fraction))
      total_Specific_heat_Based_fraction_max.append(np.max(Specific_heat_List_Based_on_fraction))
      total_Specific_heat_Based_fraction_min.append(np.min(Specific_heat_List_Based_on_fraction))
      total_Specific_heat_Based_fraction_range.append(np.ptp(Specific_heat_List_Based_on_fraction))
      total_Specific_heat_Based_fraction_std.append(np.std(Specific_heat_List_Based_on_fraction)) 
      # Average_deviation calculate
      mean_value = np.mean(Specific_heat_List_Based_on_fraction)
      res=[]
      for ele in Specific_heat_List_Based_on_fraction:
        res.append(abs(ele - mean_value))
      total_Specific_heat_Based_fraction_Ave_dev.append(np.mean(res))
      total_Specific_heat_Based_Elemental_mean.append(np.mean(Specific_heat_List))
      total_Specific_heat_Based_Elemental_median.append(np.median(Specific_heat_List))
      total_Specific_heat_Based_Elemental_variance.append(np.var(Specific_heat_List))
      total_Specific_heat_Based_Elemental_max.append(np.max(Specific_heat_List))
      total_Specific_heat_Based_Elemental_min.append(np.min(Specific_heat_List))
      total_Specific_heat_Based_Elemental_range.append(np.ptp(Specific_heat_List))
      total_Specific_heat_Based_Elemental_std.append(np.std(Specific_heat_List)) 
      # Average_deviation calculate
      mean_value = np.mean(Specific_heat_List)
      res=[]
      for ele in Specific_heat_List:
        res.append(abs(ele - mean_value))
      total_Specific_heat_Based_Elemental_Ave_dev.append(np.mean(res))
      total_Specific_heat_Based_Subscript_mean.append(np.mean(elem_Specific_heat_Subscript_List))
      total_Specific_heat_Based_Subscript_median.append(np.median(elem_Specific_heat_Subscript_List))
      total_Specific_heat_Based_Subscript_variance.append(np.var(elem_Specific_heat_Subscript_List))
      total_Specific_heat_Based_Subscript_max.append(np.max(elem_Specific_heat_Subscript_List))
      total_Specific_heat_Based_Subscript_min.append(np.min(elem_Specific_heat_Subscript_List))
      total_Specific_heat_Based_Subscript_range.append(np.ptp(elem_Specific_heat_Subscript_List))
      total_Specific_heat_Based_Subscript_std.append(np.std(elem_Specific_heat_Subscript_List)) 
      # Average_deviation calculate
      mean_value = np.mean(elem_Specific_heat_Subscript_List)
      res=[]
      for ele in elem_Specific_heat_Subscript_List:
        res.append(abs(ele - mean_value))
      total_Specific_heat_Based_Subscript_Ave_dev.append(np.mean(res))

      total_Ionic_Radius_Based_Elemental_mean.append(np.mean(Ionic_Radius_List))
      total_Ionic_Radius_Based_Elemental_median.append(np.median(Ionic_Radius_List))
      total_Ionic_Radius_Based_Elemental_variance.append(np.var(Ionic_Radius_List))
      total_Ionic_Radius_Based_Elemental_max.append(np.max(Ionic_Radius_List))
      total_Ionic_Radius_Based_Elemental_min.append(np.min(Ionic_Radius_List))
      total_Ionic_Radius_Based_Elemental_range.append(np.ptp(Ionic_Radius_List))
      total_Ionic_Radius_Based_Elemental_std.append(np.std(Ionic_Radius_List)) 
      # Average_deviation calculate
      mean_value = np.mean(Ionic_Radius_List)
      res=[]
      for ele in Ionic_Radius_List:
        res.append(abs(ele - mean_value))
      total_Ionic_Radius_Based_Elemental_Ave_dev.append(np.mean(res))

  DataSet_for_Generation_322Features['mean_ElecGativ_Fraction']=total_electronegativ_Based_fraction_mean
  DataSet_for_Generation_322Features['median_ElecGativ_Fraction']=total_electronegativ_Based_fraction_median
  DataSet_for_Generation_322Features['variance_ElecGativ_Fraction']=total_electronegativ_Based_fraction_variance
  DataSet_for_Generation_322Features['max_ElecGativ_Fraction']=total_electronegativ_Based_fraction_max
  DataSet_for_Generation_322Features['min_ElecGativ_Fraction']=total_electronegativ_Based_fraction_min
  DataSet_for_Generation_322Features['range_ElecGativ_Fraction']=total_electronegativ_Based_fraction_range
  DataSet_for_Generation_322Features['stan_dev_ElecGativ_Fraction']=total_electronegativ_Based_fraction_std
  DataSet_for_Generation_322Features['Ave_dev_ElecGativ_Fraction']=total_electronegativ_Based_fraction_Ave_dev
  DataSet_for_Generation_322Features['mean_ElecGativ_Elemental']=total_electronegativ_Based_Elemental_mean
  DataSet_for_Generation_322Features['median_ElecGativ_Elemental']=total_electronegativ_Based_Elemental_median
  DataSet_for_Generation_322Features['variance_ElecGativ_Elemental']=total_electronegativ_Based_Elemental_variance
  DataSet_for_Generation_322Features['max_ElecGativ_Elemental']=total_electronegativ_Based_Elemental_max
  DataSet_for_Generation_322Features['min_ElecGativ_Elemental']=total_electronegativ_Based_Elemental_min
  DataSet_for_Generation_322Features['range_ElecGativ_Elemental']=total_electronegativ_Based_Elemental_range
  DataSet_for_Generation_322Features['stan_dev_ElecGativ_Elemental']=total_electronegativ_Based_Elemental_std
  DataSet_for_Generation_322Features['Ave_dev_ElecGativ_Elemental']=total_electronegativ_Based_Elemental_Ave_dev
  DataSet_for_Generation_322Features['mean_ElecGativ_Subscript']=total_electronegativ_Based_Subscript_mean
  DataSet_for_Generation_322Features['median_ElecGativ_Subscript']=total_electronegativ_Based_Subscript_median
  DataSet_for_Generation_322Features['variance_ElecGativ_Subscript']=total_electronegativ_Based_Subscript_variance
  DataSet_for_Generation_322Features['max_ElecGativ_Subscript']=total_electronegativ_Based_Subscript_max
  DataSet_for_Generation_322Features['min_ElecGativ_Subscript']=total_electronegativ_Based_Subscript_min
  DataSet_for_Generation_322Features['range_ElecGativ_Subscript']=total_electronegativ_Based_Subscript_range
  DataSet_for_Generation_322Features['stan_dev_ElecGativ_Subscript']=total_electronegativ_Based_Subscript_std
  DataSet_for_Generation_322Features['Ave_dev_ElecGativ_Subscript']=total_electronegativ_Based_Subscript_Ave_dev

  DataSet_for_Generation_322Features['mean_Pettifor_fraction']=total_pettifor_number_Based_fraction_mean
  DataSet_for_Generation_322Features['median_Pettifor_fraction']=total_pettifor_number_Based_fraction_median
  DataSet_for_Generation_322Features['variance_Pettifor_fraction']=total_pettifor_number_Based_fraction_variance
  DataSet_for_Generation_322Features['max_Pettifor_fraction']=total_pettifor_number_Based_fraction_max
  DataSet_for_Generation_322Features['min_Pettifor_fraction']=total_pettifor_number_Based_fraction_min
  DataSet_for_Generation_322Features['range_Pettifor_fraction']=total_pettifor_number_Based_fraction_range
  DataSet_for_Generation_322Features['stan_dev_Pettifor_fraction']=total_pettifor_number_Based_fraction_std
  DataSet_for_Generation_322Features['Ave_dev_Pettifor_fraction']=total_pettifor_number_Based_fraction_Ave_dev
  DataSet_for_Generation_322Features['mean_Pettifor_Elemental']=total_pettifor_number_Based_Elemental_mean
  DataSet_for_Generation_322Features['median_Pettifor_Elemental']=total_pettifor_number_Based_Elemental_median
  DataSet_for_Generation_322Features['variance_Pettifor_Elemental']=total_pettifor_number_Based_Elemental_variance
  DataSet_for_Generation_322Features['max_Pettifor_Elemental']=total_pettifor_number_Based_Elemental_max
  DataSet_for_Generation_322Features['min_Pettifor_Elemental']=total_pettifor_number_Based_Elemental_min
  DataSet_for_Generation_322Features['range_Pettifor_Elemental']=total_pettifor_number_Based_Elemental_range
  DataSet_for_Generation_322Features['stan_dev_Pettifor_Elemental']=total_pettifor_number_Based_Elemental_std
  DataSet_for_Generation_322Features['Ave_dev_Pettifor_Elemental']=total_pettifor_number_Based_Elemental_Ave_dev
  DataSet_for_Generation_322Features['mean_Pettifor_Subscript']=total_pettifor_number_Based_Subscript_mean
  DataSet_for_Generation_322Features['median_Pettifor_Subscript']=total_pettifor_number_Based_Subscript_median
  DataSet_for_Generation_322Features['variance_Pettifor_Subscript']=total_pettifor_number_Based_Subscript_variance
  DataSet_for_Generation_322Features['max_Pettifor_Subscript']=total_pettifor_number_Based_Subscript_max
  DataSet_for_Generation_322Features['min_Pettifor_Subscript']=total_pettifor_number_Based_Subscript_min
  DataSet_for_Generation_322Features['range_Pettifor_Subscript']=total_pettifor_number_Based_Subscript_range
  DataSet_for_Generation_322Features['stan_dev_Pettifor_Subscript']=total_pettifor_number_Based_Subscript_std
  DataSet_for_Generation_322Features['Ave_dev_Pettifor_Subscript']=total_pettifor_number_Based_Subscript_Ave_dev

  DataSet_for_Generation_322Features['mean_VanderwaalsR_Elemental']=total_VanderwaalsR_Based_Elemental_mean
  DataSet_for_Generation_322Features['median_VanderwaalsR_Elemental']=total_VanderwaalsR_Based_Elemental_median
  DataSet_for_Generation_322Features['variance_VanderwaalsR_Elemental']=total_VanderwaalsR_Based_Elemental_variance
  DataSet_for_Generation_322Features['max_VanderwaalsR_Elemental']=total_VanderwaalsR_Based_Elemental_max
  DataSet_for_Generation_322Features['min_VanderwaalsR_Elemental']=total_VanderwaalsR_Based_Elemental_min
  DataSet_for_Generation_322Features['range_VanderwaalsR_Elemental']=total_VanderwaalsR_Based_Elemental_range
  DataSet_for_Generation_322Features['stan_dev_VanderwaalsR_Elemental']=total_VanderwaalsR_Based_Elemental_std
  DataSet_for_Generation_322Features['Ave_dev_VanderwaalsR_Elemental']=total_VanderwaalsR_Based_Elemental_Ave_dev

  DataSet_for_Generation_322Features['mean_period_Elemental']=total_period_number_Based_Elemental_mean
  DataSet_for_Generation_322Features['median_period_Elemental']=total_period_number_Based_Elemental_median
  DataSet_for_Generation_322Features['variance_period_Elemental']=total_period_number_Based_Elemental_variance
  DataSet_for_Generation_322Features['max_period_Elemental']=total_period_number_Based_Elemental_max
  DataSet_for_Generation_322Features['min_period_Elemental']=total_period_number_Based_Elemental_min
  DataSet_for_Generation_322Features['range_period_Elemental']=total_period_number_Based_Elemental_range
  DataSet_for_Generation_322Features['stan_dev_period_Elemental']=total_period_number_Based_Elemental_std
  DataSet_for_Generation_322Features['Ave_dev_period_Elemental']=total_period_number_Based_Elemental_Ave_dev

  DataSet_for_Generation_322Features['mean_group_Elemental']=total_group_number_Based_Elemental_mean
  DataSet_for_Generation_322Features['median_group_Elemental']=total_group_number_Based_Elemental_median
  DataSet_for_Generation_322Features['variance_group_Elemental']=total_group_number_Based_Elemental_variance
  DataSet_for_Generation_322Features['max_group_Elemental']=total_group_number_Based_Elemental_max
  DataSet_for_Generation_322Features['min_group_Elemental']=total_group_number_Based_Elemental_min
  DataSet_for_Generation_322Features['range_group_Elemental']=total_group_number_Based_Elemental_range
  DataSet_for_Generation_322Features['stan_dev_group_Elemental']=total_group_number_Based_Elemental_std
  DataSet_for_Generation_322Features['Ave_dev_group_Elemental']=total_group_number_Based_Elemental_Ave_dev

  DataSet_for_Generation_322Features['mean_heat_formation_fraction']=total_heat_formation_Based_fraction_mean
  DataSet_for_Generation_322Features['median_heat_formation_fraction']=total_heat_formation_Based_fraction_median
  DataSet_for_Generation_322Features['variance_heat_formation_fraction']=total_heat_formation_Based_fraction_variance
  DataSet_for_Generation_322Features['max_heat_formation_fraction']=total_heat_formation_Based_fraction_max
  DataSet_for_Generation_322Features['min_heat_formation_fraction']=total_heat_formation_Based_fraction_min
  DataSet_for_Generation_322Features['range_heat_formation_fraction']=total_heat_formation_Based_fraction_range
  DataSet_for_Generation_322Features['stan_dev_heat_formation_fraction']=total_heat_formation_Based_fraction_std
  DataSet_for_Generation_322Features['Ave_dev_heat_formation_fraction']=total_heat_formation_Based_fraction_Ave_dev
  DataSet_for_Generation_322Features['mean_heat_formation_Elemental']=total_heat_formation_Based_Elemental_mean
  DataSet_for_Generation_322Features['median_heat_formation_Elemental']=total_heat_formation_Based_Elemental_median
  DataSet_for_Generation_322Features['variance_heat_formation_Elemental']=total_heat_formation_Based_Elemental_variance
  DataSet_for_Generation_322Features['max_heat_formation_Elemental']=total_heat_formation_Based_Elemental_max
  DataSet_for_Generation_322Features['min_heat_formation_Elemental']=total_heat_formation_Based_Elemental_min
  DataSet_for_Generation_322Features['range_heat_formation_Elemental']=total_heat_formation_Based_Elemental_range
  DataSet_for_Generation_322Features['stan_dev_heat_formation_Elemental']=total_heat_formation_Based_Elemental_std
  DataSet_for_Generation_322Features['Ave_dev_heat_formation_Elemental']=total_heat_formation_Based_Elemental_Ave_dev
  DataSet_for_Generation_322Features['mean_heat_formation_Subscript']=total_heat_formation_Based_Subscript_mean
  DataSet_for_Generation_322Features['median_heat_formation_Subscript']=total_heat_formation_Based_Subscript_median
  DataSet_for_Generation_322Features['variance_heat_formation_Subscript']=total_heat_formation_Based_Subscript_variance
  DataSet_for_Generation_322Features['max_heat_formation_Subscript']=total_heat_formation_Based_Subscript_max
  DataSet_for_Generation_322Features['min_heat_formation_Subscript']=total_heat_formation_Based_Subscript_min
  DataSet_for_Generation_322Features['range_heat_formation_Subscript']=total_heat_formation_Based_Subscript_range
  DataSet_for_Generation_322Features['stan_dev_heat_formation_Subscript']=total_heat_formation_Based_Subscript_std
  DataSet_for_Generation_322Features['Ave_dev_heat_formation_Subscript']=total_heat_formation_Based_Subscript_Ave_dev

  DataSet_for_Generation_322Features['mean_UnpairedElec_fraction']=total_unpaired_electron_Based_fraction_mean
  DataSet_for_Generation_322Features['median_UnpairedElec_fraction']=total_unpaired_electron_Based_fraction_median
  DataSet_for_Generation_322Features['variance_UnpairedElec_fraction']=total_unpaired_electron_Based_fraction_variance
  DataSet_for_Generation_322Features['max_UnpairedElec_fraction']=total_unpaired_electron_Based_fraction_max
  DataSet_for_Generation_322Features['min_UnpairedElec_fraction']=total_unpaired_electron_Based_fraction_min
  DataSet_for_Generation_322Features['range_UnpairedElec_fraction']=total_unpaired_electron_Based_fraction_range
  DataSet_for_Generation_322Features['stan_dev_UnpairedElec_fraction']=total_unpaired_electron_Based_fraction_std
  DataSet_for_Generation_322Features['Ave_dev_UnpairedElec_fraction']=total_unpaired_electron_Based_fraction_Ave_dev
  DataSet_for_Generation_322Features['mean_UnpairedElec_Elemental']=total_unpaired_electron_Based_Elemental_mean
  DataSet_for_Generation_322Features['median_UnpairedElec_Elemental']=total_unpaired_electron_Based_Elemental_median
  DataSet_for_Generation_322Features['variance_UnpairedElec_Elemental']=total_unpaired_electron_Based_Elemental_variance
  DataSet_for_Generation_322Features['max_UnpairedElec_Elemental']=total_unpaired_electron_Based_Elemental_max
  DataSet_for_Generation_322Features['min_UnpairedElec_Elemental']=total_unpaired_electron_Based_Elemental_min
  DataSet_for_Generation_322Features['range_UnpairedElec_Elemental']=total_unpaired_electron_Based_Elemental_range
  DataSet_for_Generation_322Features['stan_dev_UnpairedElec_Elemental']=total_unpaired_electron_Based_Elemental_std
  DataSet_for_Generation_322Features['Ave_dev_UnpairedElec_Elemental']=total_unpaired_electron_Based_Elemental_Ave_dev
  DataSet_for_Generation_322Features['mean_UnpairedElec_Subscript']=total_unpaired_electron_Based_Subscript_mean
  DataSet_for_Generation_322Features['median_UnpairedElec_Subscript']=total_unpaired_electron_Based_Subscript_median
  DataSet_for_Generation_322Features['variance_UnpairedElec_Subscript']=total_unpaired_electron_Based_Subscript_variance
  DataSet_for_Generation_322Features['max_UnpairedElec_Subscript']=total_unpaired_electron_Based_Subscript_max
  DataSet_for_Generation_322Features['min_UnpairedElec_Subscript']=total_unpaired_electron_Based_Subscript_min
  DataSet_for_Generation_322Features['range_UnpairedElec_Subscript']=total_unpaired_electron_Based_Subscript_range
  DataSet_for_Generation_322Features['stan_dev_UnpairedElec_Subscript']=total_unpaired_electron_Based_Subscript_std
  DataSet_for_Generation_322Features['Ave_dev_UnpairedElec_Subscript']=total_unpaired_electron_Based_Subscript_Ave_dev

  DataSet_for_Generation_322Features['mean_NumElecValence_fraction']=total_Number_Elec_Valence_Based_fraction_mean
  DataSet_for_Generation_322Features['median_NumElecValence_fraction']=total_Number_Elec_Valence_Based_fraction_median
  DataSet_for_Generation_322Features['variance_NumElecValence_fraction']=total_Number_Elec_Valence_Based_fraction_variance
  DataSet_for_Generation_322Features['max_NumElecValence_fraction']=total_Number_Elec_Valence_Based_fraction_max
  DataSet_for_Generation_322Features['min_NumElecValence_fraction']=total_Number_Elec_Valence_Based_fraction_min
  DataSet_for_Generation_322Features['range_NumElecValence_fraction']=total_Number_Elec_Valence_Based_fraction_range
  DataSet_for_Generation_322Features['stan_dev_NumElecValence_fraction']=total_Number_Elec_Valence_Based_fraction_std
  DataSet_for_Generation_322Features['Ave_dev_NumElecValence_fraction']=total_Number_Elec_Valence_Based_fraction_Ave_dev
  DataSet_for_Generation_322Features['mean_NumElecValence_Elemental']=total_Number_Elec_Valence_Based_Elemental_mean
  DataSet_for_Generation_322Features['median_NumElecValence_Elemental']=total_Number_Elec_Valence_Based_Elemental_median
  DataSet_for_Generation_322Features['variance_NumElecValence_Elemental']=total_Number_Elec_Valence_Based_Elemental_variance
  DataSet_for_Generation_322Features['max_NumElecValence_Elemental']=total_Number_Elec_Valence_Based_Elemental_max
  DataSet_for_Generation_322Features['min_NumElecValence_Elemental']=total_Number_Elec_Valence_Based_Elemental_min
  DataSet_for_Generation_322Features['range_NumElecValence_Elemental']=total_Number_Elec_Valence_Based_Elemental_range
  DataSet_for_Generation_322Features['stan_dev_NumElecValence_Elemental']=total_Number_Elec_Valence_Based_Elemental_std
  DataSet_for_Generation_322Features['Ave_dev_NumElecValence_Elemental']=total_Number_Elec_Valence_Based_Elemental_Ave_dev
  DataSet_for_Generation_322Features['mean_NumElecValence_Subscript']=total_Number_Elec_Valence_Based_Subscript_mean
  DataSet_for_Generation_322Features['median_NumElecValence_Subscript']=total_Number_Elec_Valence_Based_Subscript_median
  DataSet_for_Generation_322Features['variance_NumElecValence_Subscript']=total_Number_Elec_Valence_Based_Subscript_variance
  DataSet_for_Generation_322Features['max_NumElecValence_Subscript']=total_Number_Elec_Valence_Based_Subscript_max
  DataSet_for_Generation_322Features['min_NumElecValence_Subscript']=total_Number_Elec_Valence_Based_Subscript_min
  DataSet_for_Generation_322Features['range_NumElecValence_Subscript']=total_Number_Elec_Valence_Based_Subscript_range
  DataSet_for_Generation_322Features['stan_dev_NumElecValence_Subscript']=total_Number_Elec_Valence_Based_Subscript_std
  DataSet_for_Generation_322Features['Ave_dev_NumElecValence_Subscript']=total_Number_Elec_Valence_Based_Subscript_Ave_dev

  DataSet_for_Generation_322Features['mean_MagMoment_fraction']=total_MagMoment_Based_fraction_mean
  DataSet_for_Generation_322Features['median_MagMoment_fraction']=total_MagMoment_Based_fraction_median
  DataSet_for_Generation_322Features['variance_MagMoment_fraction']=total_MagMoment_Based_fraction_variance
  DataSet_for_Generation_322Features['max_MagMoment_fraction']=total_MagMoment_Based_fraction_max
  DataSet_for_Generation_322Features['min_MagMoment_fraction']=total_MagMoment_Based_fraction_min
  DataSet_for_Generation_322Features['range_MagMoment_fraction']=total_MagMoment_Based_fraction_range
  DataSet_for_Generation_322Features['stan_dev_MagMoment_fraction']=total_MagMoment_Based_fraction_std
  DataSet_for_Generation_322Features['Ave_dev_MagMoment_fraction']=total_MagMoment_Based_fraction_Ave_dev
  DataSet_for_Generation_322Features['mean_MagMoment_Elemental']=total_MagMoment_Based_Elemental_mean
  DataSet_for_Generation_322Features['median_MagMoment_Elemental']=total_MagMoment_Based_Elemental_median
  DataSet_for_Generation_322Features['variance_MagMoment_Elemental']=total_MagMoment_Based_Elemental_variance
  DataSet_for_Generation_322Features['max_MagMoment_Elemental']=total_MagMoment_Based_Elemental_max
  DataSet_for_Generation_322Features['min_MagMoment_Elemental']=total_MagMoment_Based_Elemental_min
  DataSet_for_Generation_322Features['range_MagMoment_Elemental']=total_MagMoment_Based_Elemental_range
  DataSet_for_Generation_322Features['stan_dev_MagMoment_Elemental']=total_MagMoment_Based_Elemental_std
  DataSet_for_Generation_322Features['Ave_dev_MagMoment_Elemental']=total_MagMoment_Based_Elemental_Ave_dev
  DataSet_for_Generation_322Features['mean_MagMoment_Subscript']=total_MagMoment_Based_Subscript_mean
  DataSet_for_Generation_322Features['median_MagMoment_Subscript']=total_MagMoment_Based_Subscript_median
  DataSet_for_Generation_322Features['variance_MagMoment_Subscript']=total_MagMoment_Based_Subscript_variance
  DataSet_for_Generation_322Features['max_MagMoment_Subscript']=total_MagMoment_Based_Subscript_max
  DataSet_for_Generation_322Features['min_MagMoment_Subscript']=total_MagMoment_Based_Subscript_min
  DataSet_for_Generation_322Features['range_MagMoment_Subscript']=total_MagMoment_Based_Subscript_range
  DataSet_for_Generation_322Features['stan_dev_MagMoment_Subscript']=total_MagMoment_Based_Subscript_std
  DataSet_for_Generation_322Features['Ave_dev_MagMoment_Subscript']=total_MagMoment_Based_Subscript_Ave_dev

  DataSet_for_Generation_322Features['mean_dipole_polarizability_fraction']=total_dipole_polarizability_Based_fraction_mean
  DataSet_for_Generation_322Features['median_dipole_polarizability_fraction']=total_dipole_polarizability_Based_fraction_median
  DataSet_for_Generation_322Features['variance_dipole_polarizability_fraction']=total_dipole_polarizability_Based_fraction_variance
  DataSet_for_Generation_322Features['max_dipole_polarizability_fraction']=total_dipole_polarizability_Based_fraction_max
  DataSet_for_Generation_322Features['min_dipole_polarizability_fraction']=total_dipole_polarizability_Based_fraction_min
  DataSet_for_Generation_322Features['range_dipole_polarizability_fraction']=total_dipole_polarizability_Based_fraction_range
  DataSet_for_Generation_322Features['stan_dev_dipole_polarizability_fraction']=total_dipole_polarizability_Based_fraction_std
  DataSet_for_Generation_322Features['Ave_dev_dipole_polarizability_fraction']=total_dipole_polarizability_Based_fraction_Ave_dev
  DataSet_for_Generation_322Features['mean_dipole_polarizability_Elemental']=total_dipole_polarizability_Based_Elemental_mean
  DataSet_for_Generation_322Features['median_dipole_polarizability_Elemental']=total_dipole_polarizability_Based_Elemental_median
  DataSet_for_Generation_322Features['variance_dipole_polarizability_Elemental']=total_dipole_polarizability_Based_Elemental_variance
  DataSet_for_Generation_322Features['max_dipole_polarizability_Elemental']=total_dipole_polarizability_Based_Elemental_max
  DataSet_for_Generation_322Features['min_dipole_polarizability_Elemental']=total_dipole_polarizability_Based_Elemental_min
  DataSet_for_Generation_322Features['range_dipole_polarizability_Elemental']=total_dipole_polarizability_Based_Elemental_range
  DataSet_for_Generation_322Features['stan_dev_dipole_polarizability_Elemental']=total_dipole_polarizability_Based_Elemental_std
  DataSet_for_Generation_322Features['Ave_dev_dipole_polarizability_Elemental']=total_dipole_polarizability_Based_Elemental_Ave_dev
  DataSet_for_Generation_322Features['mean_dipole_polarizability_Subscript']=total_dipole_polarizability_Based_Subscript_mean
  DataSet_for_Generation_322Features['median_dipole_polarizability_Subscript']=total_dipole_polarizability_Based_Subscript_median
  DataSet_for_Generation_322Features['variance_dipole_polarizability_Subscript']=total_dipole_polarizability_Based_Subscript_variance
  DataSet_for_Generation_322Features['max_dipole_polarizability_Subscript']=total_dipole_polarizability_Based_Subscript_max
  DataSet_for_Generation_322Features['min_dipole_polarizability_Subscript']=total_dipole_polarizability_Based_Subscript_min
  DataSet_for_Generation_322Features['range_dipole_polarizability_Subscript']=total_dipole_polarizability_Based_Subscript_range
  DataSet_for_Generation_322Features['stan_dev_dipole_polarizability_Subscript']=total_dipole_polarizability_Based_Subscript_std
  DataSet_for_Generation_322Features['Ave_dev_dipole_polarizability_Subscript']=total_dipole_polarizability_Based_Subscript_Ave_dev

  DataSet_for_Generation_322Features['mean_First_Ionis_Energy_fraction']=total_First_Ionis_Energy_Based_fraction_mean
  DataSet_for_Generation_322Features['median_First_Ionis_Energy_fraction']=total_First_Ionis_Energy_Based_fraction_median
  DataSet_for_Generation_322Features['variance_First_Ionis_Energy_fraction']=total_First_Ionis_Energy_Based_fraction_variance
  DataSet_for_Generation_322Features['max_First_Ionis_Energy_fraction']=total_First_Ionis_Energy_Based_fraction_max
  DataSet_for_Generation_322Features['min_First_Ionis_Energy_fraction']=total_First_Ionis_Energy_Based_fraction_min
  DataSet_for_Generation_322Features['range_First_Ionis_Energy_fraction']=total_First_Ionis_Energy_Based_fraction_range
  DataSet_for_Generation_322Features['stan_dev_First_Ionis_Energy_fraction']=total_First_Ionis_Energy_Based_fraction_std
  DataSet_for_Generation_322Features['Ave_dev_First_Ionis_Energy_fraction']=total_First_Ionis_Energy_Based_fraction_Ave_dev
  DataSet_for_Generation_322Features['mean_First_Ionis_Energy_Elemental']=total_First_Ionis_Energy_Based_Elemental_mean
  DataSet_for_Generation_322Features['median_First_Ionis_Energy_Elemental']=total_First_Ionis_Energy_Based_Elemental_median
  DataSet_for_Generation_322Features['variance_First_Ionis_Energy_Elemental']=total_First_Ionis_Energy_Based_Elemental_variance
  DataSet_for_Generation_322Features['max_First_Ionis_Energy_Elemental']=total_First_Ionis_Energy_Based_Elemental_max
  DataSet_for_Generation_322Features['min_First_Ionis_Energy_Elemental']=total_First_Ionis_Energy_Based_Elemental_min
  DataSet_for_Generation_322Features['range_First_Ionis_Energy_Elemental']=total_First_Ionis_Energy_Based_Elemental_range
  DataSet_for_Generation_322Features['stan_dev_First_Ionis_Energy_Elemental']=total_First_Ionis_Energy_Based_Elemental_std
  DataSet_for_Generation_322Features['Ave_dev_First_Ionis_Energy_Elemental']=total_First_Ionis_Energy_Based_Elemental_Ave_dev
  DataSet_for_Generation_322Features['mean_First_Ionis_Energy_Subscript']=total_First_Ionis_Energy_Based_Subscript_mean
  DataSet_for_Generation_322Features['median_First_Ionis_Energy_Subscript']=total_First_Ionis_Energy_Based_Subscript_median
  DataSet_for_Generation_322Features['variance_First_Ionis_Energy_Subscript']=total_First_Ionis_Energy_Based_Subscript_variance
  DataSet_for_Generation_322Features['max_First_Ionis_Energy_Subscript']=total_First_Ionis_Energy_Based_Subscript_max
  DataSet_for_Generation_322Features['min_First_Ionis_Energy_Subscript']=total_First_Ionis_Energy_Based_Subscript_min
  DataSet_for_Generation_322Features['range_First_Ionis_Energy_Subscript']=total_First_Ionis_Energy_Based_Subscript_range
  DataSet_for_Generation_322Features['stan_dev_First_Ionis_Energy_Subscript']=total_First_Ionis_Energy_Based_Subscript_std
  DataSet_for_Generation_322Features['Ave_dev_First_Ionis_Energy_Subscript']=total_First_Ionis_Energy_Based_Subscript_Ave_dev

  DataSet_for_Generation_322Features['mean_ElecAffinity_fraction']=total_ElecAffinity_Based_fraction_mean
  DataSet_for_Generation_322Features['median_ElecAffinity_fraction']=total_ElecAffinity_Based_fraction_median
  DataSet_for_Generation_322Features['variance_ElecAffinity_fraction']=total_ElecAffinity_Based_fraction_variance
  DataSet_for_Generation_322Features['max_ElecAffinity_fraction']=total_ElecAffinity_Based_fraction_max
  DataSet_for_Generation_322Features['min_ElecAffinity_fraction']=total_ElecAffinity_Based_fraction_min
  DataSet_for_Generation_322Features['range_ElecAffinity_fraction']=total_ElecAffinity_Based_fraction_range
  DataSet_for_Generation_322Features['stan_dev_ElecAffinity_fraction']=total_ElecAffinity_Based_fraction_std
  DataSet_for_Generation_322Features['Ave_dev_ElecAffinity_fraction']=total_ElecAffinity_Based_fraction_Ave_dev
  DataSet_for_Generation_322Features['mean_ElecAffinity_Elemental']=total_ElecAffinity_Based_Elemental_mean
  DataSet_for_Generation_322Features['median_ElecAffinity_Elemental']=total_ElecAffinity_Based_Elemental_median
  DataSet_for_Generation_322Features['variance_ElecAffinity_Elemental']=total_ElecAffinity_Based_Elemental_variance
  DataSet_for_Generation_322Features['max_ElecAffinity_Elemental']=total_ElecAffinity_Based_Elemental_max
  DataSet_for_Generation_322Features['min_ElecAffinity_Elemental']=total_ElecAffinity_Based_Elemental_min
  DataSet_for_Generation_322Features['range_ElecAffinity_Elemental']=total_ElecAffinity_Based_Elemental_range
  DataSet_for_Generation_322Features['stan_dev_ElecAffinity_Elemental']=total_ElecAffinity_Based_Elemental_std
  DataSet_for_Generation_322Features['Ave_dev_ElecAffinity_Elemental']=total_ElecAffinity_Based_Elemental_Ave_dev
  DataSet_for_Generation_322Features['mean_ElecAffinity_Subscript']=total_ElecAffinity_Based_Subscript_mean
  DataSet_for_Generation_322Features['median_ElecAffinity_Subscript']=total_ElecAffinity_Based_Subscript_median
  DataSet_for_Generation_322Features['variance_ElecAffinity_Subscript']=total_ElecAffinity_Based_Subscript_variance
  DataSet_for_Generation_322Features['max_ElecAffinity_Subscript']=total_ElecAffinity_Based_Subscript_max
  DataSet_for_Generation_322Features['min_ElecAffinity_Subscript']=total_ElecAffinity_Based_Subscript_min
  DataSet_for_Generation_322Features['range_ElecAffinity_Subscript']=total_ElecAffinity_Based_Subscript_range
  DataSet_for_Generation_322Features['stan_dev_ElecAffinity_Subscript']=total_ElecAffinity_Based_Subscript_std
  DataSet_for_Generation_322Features['Ave_dev_ElecAffinity_Subscript']=total_ElecAffinity_Based_Subscript_Ave_dev

  DataSet_for_Generation_322Features['mean_Thermal_conduct_fraction']=total_Thermal_conduct_Based_fraction_mean
  DataSet_for_Generation_322Features['median_Thermal_conduct_fraction']=total_Thermal_conduct_Based_fraction_median
  DataSet_for_Generation_322Features['variance_Thermal_conduct_fraction']=total_Thermal_conduct_Based_fraction_variance
  DataSet_for_Generation_322Features['max_Thermal_conduct_fraction']=total_Thermal_conduct_Based_fraction_max
  DataSet_for_Generation_322Features['min_Thermal_conduct_fraction']=total_Thermal_conduct_Based_fraction_min
  DataSet_for_Generation_322Features['range_Thermal_conduct_fraction']=total_Thermal_conduct_Based_fraction_range
  DataSet_for_Generation_322Features['stan_dev_Thermal_conduct_fraction']=total_Thermal_conduct_Based_fraction_std
  DataSet_for_Generation_322Features['Ave_dev_Thermal_conduct_fraction']=total_Thermal_conduct_Based_fraction_Ave_dev
  DataSet_for_Generation_322Features['mean_Thermal_conduct_Elemental']=total_Thermal_conduct_Based_Elemental_mean
  DataSet_for_Generation_322Features['median_Thermal_conduct_Elemental']=total_Thermal_conduct_Based_Elemental_median
  DataSet_for_Generation_322Features['variance_Thermal_conduct_Elemental']=total_Thermal_conduct_Based_Elemental_variance
  DataSet_for_Generation_322Features['max_Thermal_conduct_Elemental']=total_Thermal_conduct_Based_Elemental_max
  DataSet_for_Generation_322Features['min_Thermal_conduct_Elemental']=total_Thermal_conduct_Based_Elemental_min
  DataSet_for_Generation_322Features['range_Thermal_conduct_Elemental']=total_Thermal_conduct_Based_Elemental_range
  DataSet_for_Generation_322Features['stan_dev_Thermal_conduct_Elemental']=total_Thermal_conduct_Based_Elemental_std
  DataSet_for_Generation_322Features['Ave_dev_Thermal_conduct_Elemental']=total_Thermal_conduct_Based_Elemental_Ave_dev
  DataSet_for_Generation_322Features['mean_Thermal_conduct_Subscript']=total_Thermal_conduct_Based_Subscript_mean
  DataSet_for_Generation_322Features['median_Thermal_conduct_Subscript']=total_Thermal_conduct_Based_Subscript_median
  DataSet_for_Generation_322Features['variance_Thermal_conduct_Subscript']=total_Thermal_conduct_Based_Subscript_variance
  DataSet_for_Generation_322Features['max_Thermal_conduct_Subscript']=total_Thermal_conduct_Based_Subscript_max
  DataSet_for_Generation_322Features['min_Thermal_conduct_Subscript']=total_Thermal_conduct_Based_Subscript_min
  DataSet_for_Generation_322Features['range_Thermal_conduct_Subscript']=total_Thermal_conduct_Based_Subscript_range
  DataSet_for_Generation_322Features['stan_dev_Thermal_conduct_Subscript']=total_Thermal_conduct_Based_Subscript_std
  DataSet_for_Generation_322Features['Ave_dev_Thermal_conduct_Subscript']=total_Thermal_conduct_Based_Subscript_Ave_dev

  DataSet_for_Generation_322Features['mean_Electric_Conduct_fraction']=total_Electric_Conduct_Based_fraction_mean
  DataSet_for_Generation_322Features['median_Electric_Conduct_fraction']=total_Electric_Conduct_Based_fraction_median
  DataSet_for_Generation_322Features['variance_Electric_Conduct_fraction']=total_Electric_Conduct_Based_fraction_variance
  DataSet_for_Generation_322Features['max_Electric_Conduct_fraction']=total_Electric_Conduct_Based_fraction_max
  DataSet_for_Generation_322Features['min_Electric_Conduct_fraction']=total_Electric_Conduct_Based_fraction_min
  DataSet_for_Generation_322Features['range_Electric_Conduct_fraction']=total_Electric_Conduct_Based_fraction_range
  DataSet_for_Generation_322Features['stan_dev_Electric_Conduct_fraction']=total_Electric_Conduct_Based_fraction_std
  DataSet_for_Generation_322Features['Ave_dev_Electric_Conduct_fraction']=total_Electric_Conduct_Based_fraction_Ave_dev
  DataSet_for_Generation_322Features['mean_Electric_Conduct_Elemental']=total_Electric_Conduct_Based_Elemental_mean
  DataSet_for_Generation_322Features['median_Electric_Conduct_Elemental']=total_Electric_Conduct_Based_Elemental_median
  DataSet_for_Generation_322Features['variance_Electric_Conduct_Elemental']=total_Electric_Conduct_Based_Elemental_variance
  DataSet_for_Generation_322Features['max_Electric_Conduct_Elemental']=total_Electric_Conduct_Based_Elemental_max
  DataSet_for_Generation_322Features['min_Electric_Conduct_Elemental']=total_Electric_Conduct_Based_Elemental_min
  DataSet_for_Generation_322Features['range_Electric_Conduct_Elemental']=total_Electric_Conduct_Based_Elemental_range
  DataSet_for_Generation_322Features['stan_dev_Electric_Conduct_Elemental']=total_Electric_Conduct_Based_Elemental_std
  DataSet_for_Generation_322Features['Ave_dev_Electric_Conduct_Elemental']=total_Electric_Conduct_Based_Elemental_Ave_dev
  DataSet_for_Generation_322Features['mean_Electric_Conduct_Subscript']=total_Electric_Conduct_Based_Subscript_mean
  DataSet_for_Generation_322Features['median_Electric_Conduct_Subscript']=total_Electric_Conduct_Based_Subscript_median
  DataSet_for_Generation_322Features['variance_Electric_Conduct_Subscript']=total_Electric_Conduct_Based_Subscript_variance
  DataSet_for_Generation_322Features['max_Electric_Conduct_Subscript']=total_Electric_Conduct_Based_Subscript_max
  DataSet_for_Generation_322Features['min_Electric_Conduct_Subscript']=total_Electric_Conduct_Based_Subscript_min
  DataSet_for_Generation_322Features['range_Electric_Conduct_Subscript']=total_Electric_Conduct_Based_Subscript_range
  DataSet_for_Generation_322Features['stan_dev_Electric_Conduct_Subscript']=total_Electric_Conduct_Based_Subscript_std
  DataSet_for_Generation_322Features['Ave_dev_Electric_Conduct_Subscript']=total_Electric_Conduct_Based_Subscript_Ave_dev

  DataSet_for_Generation_322Features['mean_Specific_heat_fraction']=total_Specific_heat_Based_fraction_mean
  DataSet_for_Generation_322Features['median_Specific_heat_fraction']=total_Specific_heat_Based_fraction_median
  DataSet_for_Generation_322Features['variance_Specific_heat_fraction']=total_Specific_heat_Based_fraction_variance
  DataSet_for_Generation_322Features['max_Specific_heat_fraction']=total_Specific_heat_Based_fraction_max
  DataSet_for_Generation_322Features['min_Specific_heat_fraction']=total_Specific_heat_Based_fraction_min
  DataSet_for_Generation_322Features['range_Specific_heat_fraction']=total_Specific_heat_Based_fraction_range
  DataSet_for_Generation_322Features['stan_dev_Specific_heat_fraction']=total_Specific_heat_Based_fraction_std
  DataSet_for_Generation_322Features['Ave_dev_Specific_heat_fraction']=total_Specific_heat_Based_fraction_Ave_dev
  DataSet_for_Generation_322Features['mean_Specific_heat_Elemental']=total_Specific_heat_Based_Elemental_mean
  DataSet_for_Generation_322Features['median_Specific_heat_Elemental']=total_Specific_heat_Based_Elemental_median
  DataSet_for_Generation_322Features['variance_Specific_heat_Elemental']=total_Specific_heat_Based_Elemental_variance
  DataSet_for_Generation_322Features['max_Specific_heat_Elemental']=total_Specific_heat_Based_Elemental_max
  DataSet_for_Generation_322Features['min_Specific_heat_Elemental']=total_Specific_heat_Based_Elemental_min
  DataSet_for_Generation_322Features['range_Specific_heat_Elemental']=total_Specific_heat_Based_Elemental_range
  DataSet_for_Generation_322Features['stan_dev_Specific_heat_Elemental']=total_Specific_heat_Based_Elemental_std
  DataSet_for_Generation_322Features['Ave_dev_Specific_heat_Elemental']=total_Specific_heat_Based_Elemental_Ave_dev
  DataSet_for_Generation_322Features['mean_Specific_heat_Subscript']=total_Specific_heat_Based_Subscript_mean
  DataSet_for_Generation_322Features['median_Specific_heat_Subscript']=total_Specific_heat_Based_Subscript_median
  DataSet_for_Generation_322Features['variance_Specific_heat_Subscript']=total_Specific_heat_Based_Subscript_variance
  DataSet_for_Generation_322Features['max_Specific_heat_Subscript']=total_Specific_heat_Based_Subscript_max
  DataSet_for_Generation_322Features['min_Specific_heat_Subscript']=total_Specific_heat_Based_Subscript_min
  DataSet_for_Generation_322Features['range_Specific_heat_Subscript']=total_Specific_heat_Based_Subscript_range
  DataSet_for_Generation_322Features['stan_dev_Specific_heat_Subscript']=total_Specific_heat_Based_Subscript_std
  DataSet_for_Generation_322Features['Ave_dev_Specific_heat_Subscript']=total_Specific_heat_Based_Subscript_Ave_dev

  DataSet_for_Generation_322Features['mean_Ionic_Radius_Elemental']=total_Ionic_Radius_Based_Elemental_mean
  DataSet_for_Generation_322Features['median_Ionic_Radius_Elemental']=total_Ionic_Radius_Based_Elemental_median
  DataSet_for_Generation_322Features['variance_Ionic_Radius_Elemental']=total_Ionic_Radius_Based_Elemental_variance
  DataSet_for_Generation_322Features['max_Ionic_Radius_Elemental']=total_Ionic_Radius_Based_Elemental_max
  DataSet_for_Generation_322Features['min_Ionic_Radius_Elemental']=total_Ionic_Radius_Based_Elemental_min
  DataSet_for_Generation_322Features['range_Ionic_Radius_Elemental']=total_Ionic_Radius_Based_Elemental_range
  DataSet_for_Generation_322Features['stan_dev_Ionic_Radius_Elemental']=total_Ionic_Radius_Based_Elemental_std
  DataSet_for_Generation_322Features['Ave_dev_Ionic_Radius_Elemental']=total_Ionic_Radius_Based_Elemental_Ave_dev
  return DataSet_for_Generation_322Features


def save_dataframe():
    output_file_path = input("Please enter a path to save the output file in your system like the example: /home/Test ")
    output_file_name = "output.csv"  # Desired output file name
    DataSet_for_Generation_322Features.to_csv(output_file_path + "/" + output_file_name, index=False)
    print("Dataframe saved successfully.")
  

def main():
    global DataSet_for_Generation_322Features
    # Call functions
    input_file = input("Please enter the file path as in the example: home/Test/example.csv or example.xlsx or example.json ")
    # Read the input file and convert it to a DataFrame
    DataSet_for_Generation_322Features = read_input_file(input_file)
    data_preprocessing(DataSet_for_Generation_322Features)
    Generation_322_Features(DataSet_for_Generation_322Features)
    save_dataframe()
    return DataSet_for_Generation_322Features

  
main=main()   
main

