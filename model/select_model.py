"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""

def define_Model(opt):
    model = opt['model']

    if model == 'plain':
        from model.model_plain import ModelPlain
    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = ModelPlain(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
