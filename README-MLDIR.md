## 📜 The MLDir Manifesto

□ Each stage in the ML pipeline should be a separate directory.

□ For any pipeline stage, each alternative that implements that stage should be in a separate directory within that stage's folder. For instance, different models under the ModelPipelines directory.

□ A model's pipeline is defined as a composition of a processing method, an extracted feature and a model. Different model pipelines reside under the model's folder in the ModelPipelines directory. 

□ If a model's pipeline using processing method P1, features FX and the model is MY then the name of the pipeline notebook file is P1-FX-MY.ipynb. 

□ Any implementation of a stage is a set of functions and it is defined in a .py file within the directory of the specific alternative of the stage. If an implementation is small then one clearly divided Python file grouping various ones may be used.

□ Notebooks are only for demonstration or running entire pipelines (e.g., training and hyperparameter tuning). Functions are never defined in a notebook.

□ Any notebook cell should be preced with a clear heading that describes what it does.

□ Call the training, validation and testing data x_train, x_val and x_test respectively. The variable name may be appended with an _{letter} to indicate the stage of the pipeline that has produced it.

□ Whenever possible, save processed data, features or models if their computation is expensive.

□ Include a Saved directory for saving trained models, figures and other artifacts. S

□ If a pipeline stage may benefit from visualizion then provide a method for that.

□ Similar or related visualizations should be grouped together in the same figure whenever possible.

□ Logging should be implemented for every pipeline.

□ Once the experimentation phase is over (converged on a pipeline with fixed hyperparameters), the pipeline should be implemented in a single .py file with a single pipeline function. The project, except for this file can be archived at this point after producing a requirements.txt

