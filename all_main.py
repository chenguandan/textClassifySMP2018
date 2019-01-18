import hybridattmodel
import hybridconvmodel
# import hybridddualpathmodel
import hybriddensemodel
import hybriddensemamodel
import hybriddpcnnmodel
import hybridgatedconvmodel
import hybridgateddeepcnnmodel
import hybridRCNNmodel

import conditionconvmodel
import conditiondpcnnmodel
import conditiongatedconvmodel
import conditiongateddeepcnnmodel

import conditionRCNNmodel
import conditionattmodel

# def cv_main():
#     cv_num = 5
#     for cv_index in range(cv_num):
#         print('cv index', cv_index, '/', cv_num)
#         hybridconvmodel.train_model_cv(cv_index, cv_num)
#
#     # for cv_index in range(cv_num):
#     #     print('cv index', cv_index, '/', cv_num)
#     #     hybriddensemodel.train_model_cv(cv_index, cv_num)
#
#     #OOM
#     # for cv_index in range(cv_num):
#     #     print('cv index', cv_index, '/', cv_num)
#     #     hybriddensemamodel.train_model_cv(cv_index, cv_num)
#     for cv_index in range(cv_num):
#         print('cv index', cv_index, '/', cv_num)
#         hybriddpcnnmodel.train_model_cv(cv_index, cv_num)
#     for cv_index in range(cv_num):
#         print('cv index', cv_index, '/', cv_num)
#         hybridgatedconvmodel.train_model_cv(cv_index, cv_num)
#     for cv_index in range(cv_num):
#         print('cv index', cv_index, '/', cv_num)
#         hybridgateddeepcnnmodel.train_model_cv(cv_index, cv_num)
#     for cv_index in range(cv_num):
#         print('cv index', cv_index, '/', cv_num)
#         hybridRCNNmodel.train_model_cv(cv_index, cv_num)
#     # for cv_index in range(cv_num):
#     #     print('cv index', cv_index, '/', cv_num)
#     #     hybridconvmodel.train_model_cv(cv_index, cv_num)
#
# def cv_condition_main():
#     cv_num = 5
#     for cv_index in range(cv_num):
#         print('cv index', cv_index, '/', cv_num)
#         conditionconvmodel.train_model_cv(cv_index, cv_num)
#
#     for cv_index in range(1,cv_num):
#         print('cv index', cv_index, '/', cv_num)
#         conditiondpcnnmodel.train_model_cv(cv_index, cv_num)
#     for cv_index in range(cv_num):
#         print('cv index', cv_index, '/', cv_num)
#         conditiongatedconvmodel.train_model_cv(cv_index, cv_num)
#     for cv_index in range(cv_num):
#         print('cv index', cv_index, '/', cv_num)
#         conditiongateddeepcnnmodel.train_model_cv(cv_index, cv_num)
#     for cv_index in range(cv_num):
#         print('cv index', cv_index, '/', cv_num)
#         conditionattmodel.train_model_cv(cv_index, cv_num)
#     for cv_index in range(cv_num):
#         print('cv index', cv_index, '/', cv_num)
#         conditionRCNNmodel.train_model_cv(cv_index, cv_num)
#
#
# def pe_main():
#     cv_num = 5
#     # hybridconvmodel.train_model_pe()
#
#     # hybriddensemodel.train_model_pe()
#     # hybriddensemamodel.train_model_pe()
#     # hybriddpcnnmodel.train_model_pe( )
#
#     # hybridgatedconvmodel.train_model_pe()
#     # hybridgateddeepcnnmodel.train_model_pe()
#     hybridRCNNmodel.train_model_pe()
#
#     conditionconvmodel.train_model_pe()
#     conditiondpcnnmodel.train_model_pe()
#     conditiongatedconvmodel.train_model_pe()
#     conditiongateddeepcnnmodel.train_model_pe()
#     conditionattmodel.train_model_pe()
#     conditionRCNNmodel.train_model_pe()


if __name__ == '__main__':
    """usage: allmain.py pe"""
    import conditiondensemamodel
    import conditionsenetwork
    import os
    os.system('python hybridconvmodel.py pe')
    os.system('python hybriddensemamodel.py pe')
    os.system('python hybriddensemodel.py pe')
    os.system('python hybriddpcnnmodel.py pe')
    os.system('python hybridgatedconvmodel.py pe')
    os.system('python hybridgateddeepcnnmodel.py pe')
    os.system('python hybridsenetwork.py pe')
    os.system('python hybridRCNNmodel.py pe')
    os.system('python hybridattmodel.py pe')


    os.system('python conditionconvmodel.py pe')
    os.system('python conditiondensemamodel.py pe')
    os.system('python conditiondensemodel.py pe')
    os.system('python conditiondpcnnmodel.py pe')
    os.system('python conditiongatedconvmodel.py pe')
    os.system('python conditiongateddeepcnnmodel.py pe')
    os.system('python conditionsenetwork.py pe')
    os.system('python conditionRCNNmodel.py pe')
    os.system('python conditionattmodel.py pe')

    # os.system('python hybriddensemamodel.py oe')
    # os.system('python hybriddensemodel.py oe')
    # os.system('python hybriddpcnnmodel.py oe')
    # os.system('python hybridgatedconvmodel.py oe')
    # os.system('python hybridgateddeepcnnmodel.py oe')
    # os.system('python hybridsenetwork.py oe')
    # os.system('python hybridRCNNmodel.py oe')
    # os.system('python hybridattmodel.py oe')
    #
    # os.system('python conditionconvmodel.py oe')
    # os.system('python conditiondensemamodel.py oe')
    # os.system('python conditiondensemodel.py oe')
    # os.system('python conditiondpcnnmodel.py oe')
    # os.system('python conditiongatedconvmodel.py oe')
    # os.system('python conditiongateddeepcnnmodel.py oe')
    # os.system('python conditionsenetwork.py oe')
    # os.system('python conditionRCNNmodel.py oe')
    # os.system('python conditionattmodel.py oe')

    # import conditionmodelbase
    # main = conditiondensemamodel.ConditionMain('conditiondensemamodel', conditiondensemamodel.ConditionDenseMAModel)
    # main.main()
    # del main
    # main = conditionmodelbase.BaseMain('conditionsemodel', conditionsenetwork.ConditionSENetwork)
    # main.main()
    # del main
    # pe_main()
    # cv_main( )
    # cv_condition_main( )