from cframe.dataloader import DataConfiger
from cframe.dataloader import ClassificationDataloaderManager
from cframe.models import ModelConfiger
from cframe.models import ModelManager
from cframe.learner import BasicLearner


if __name__ == '__main__':
    data_config = DataConfiger.get_data_config('garbage')
    dl_manager = ClassificationDataloaderManager(data_config)

    model_config = ModelConfiger.get_model_config('simplenet')
    model_manager = ModelManager(model_config)

    learner = BasicLearner(model_manager, dl_manager,
                           optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model = learner.model
    # model.compile()
    # history = model.fit_generator(
    #     learner.train_dl,
    #     use_multiprocessing=False,
    #     validation_data=learner.valid_dl,
    #     workers=1,
    #     epochs=7
    # )
    learner.train(3)




