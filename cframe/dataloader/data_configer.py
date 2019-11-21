DATA_INFO = dict(
    CamVid=dict(dir='CamVid', n_classes=13,
                columns=['img', 'segment']),

    leaf=dict(dir='leaf-classification', n_classes=99,
              columns=['img', 'label']),
    garbage=dict(dir='garbage_classify', n_classes=40),

    DUT=dict(dir='DUT-OMRON',
             columns=['img', 'fixation', 'saliency']),

    MSRA_B=dict(dir='MSRA_B', n_classes=2,
                columns=['img', 'saliency']),

    SOC6k=dict(dir='SOC6K', n_classes=2,
               columns=['img', 'instance', 'saliency', 'sense_number']),

    InstanceSaliency1000=dict(dir='InstanceSaliency1000', n_classes=2,
                              columns=['img', 'instance', 'saliency']),

    ECSSD=dict(dir='ECSSD', n_classes=2),
    SALICON=dict(dir='SALICON',
                 columns=['img', 'saliency', 'fixation']),

    MSRA_B_InstanceSaliency1000=dict(dir='MSRA_B_InstanceSaliency1000', n_classes=2,
                                     columns=['img', 'instance', 'saliency'])
)

DATA_ROOT = dict(
    root_dir='/home/he/Data',
    dataset_dir='DataSets',
    data_name='garbage',
    csv_dir='CSVs',
    n_classes=20,
    batch_size=64,
    resize=(224, 224),
    in_channels=3,
    data_info=DATA_INFO['CamVid'],
)


class DataConfiger(object):
    @classmethod
    def set_data_root_dir(cls, data_root_dir):
        DATA_ROOT['root_dir'] = data_root_dir

    @classmethod
    def get_all_data_name(cls):
        return DATA_INFO.keys()

    @classmethod
    def get_data_config(cls, data_name):
        new_config = DATA_ROOT
        new_config['data_info'] = DATA_INFO[data_name]
        new_config['data_name'] = DATA_INFO[data_name]['dir']
        return new_config
