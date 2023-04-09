## 📜 The MLDir Manifesto
➜ Each stage in the ML pipeline should be a separate directory.

➜ If that stage is further broken down into sub-stages, each sub-stage should be a separate directory.

➜ For any pipeline stage, each alternative that implements that stage should be in a separate directory within that stage's folder.

➜ Any implementation of a stage is a set of functions.

➜ Functions are defined only in .py files not in notebooks.

➜ Notebooks are only for testing or running entire pipelines (e.g., training and hyperparameter tuning). They import the needed functions from pipeline stages.



### 📜 A more fine-grained version of the MLDir Manifesto also specifies

➜ In the implementation of any stage, any logic not specifically related to the stage's implementation such as saving or visualizion should be in a seperate function in the file.

➜ Call the training, validation and testing data x_train, x_val and x_test respectively. x_val is x_val even if x_test doesn't exist (yet). The variable name may be appended with an _{letter} to indicate the stage of the pipeline that resulted in it.

➜ Once the experimentation phase is over (converged on a pipeline with fixed hyperparameters), the pipeline should be implemented in a single .py file with a single pipeline function. The project, except for this file can be archived at this point (along with a requirements.txt).

### 📜 Which also has the following extension

➜ If a pipeline stage takes time then provide a way to save its results and load them later.

➜ If a pipeline stage may benefit from visualizion then provide a method for that.

➜ Logging should be implemented for every pipeline.


