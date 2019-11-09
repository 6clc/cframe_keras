from cframe.data_generator import DataConfiger
from cframe.data_generator import ClassificationGeneratorManager
from cframe.models import ModelConfiger
from cframe.models import ModelManager
from cframe.learner import BasicLearner


if __name__ == '__main__':
    data_config = DataConfiger.get_data_config('leaf')
    dl_manager = ClassificationGeneratorManager(data_config)

    model_config = ModelConfiger.get_model_config('simplenet')
    model_manager = ModelManager(model_config)

    learner = BasicLearner(model_manager, dl_manager)
    model = learner.model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit_generator(
        learner.train_dl,
        use_multiprocessing=False,
        validation_data=learner.valid_dl,
        workers=1,
        epochs=7
    )




