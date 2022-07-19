package org.theseed.dl4j;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.theseed.dl4j.predict.MultiRunProcessor;
import org.theseed.dl4j.predict.PredictionProcessor;
import org.theseed.dl4j.predict.ValidateProcessor;
import org.theseed.dl4j.train.ImproveProcessor;
import org.theseed.dl4j.train.RandomForestTrainProcessor;
import org.theseed.dl4j.train.RegressionTrainingProcessor;
import org.theseed.dl4j.train.RocProcessor;
import org.theseed.dl4j.train.SearchProcessor;
import org.theseed.dl4j.train.TrainCheckProcessor;
import org.theseed.dl4j.train.ClassTrainingProcessor;
import org.theseed.dl4j.train.CrossValidateProcessor;
import org.theseed.utils.ICommand;
import org.theseed.utils.Parms;
import org.theseed.utils.ParseFailureException;

/**
 * Main entry point for the Deep Learning utility.  The first parameter is a command-- use "train" to
 * train a classification model with a training set and "rtrain" to train a regression model.
 * Use "predict" to apply a model to a prediction set.  Use "search" to test multiple model configurations.
 * Use "multirun" to run predictions on multiple models in a single directory.
 *
 * If the command is followed by an equal sign, then the part after the equal sign should be a file name.
 * The parameters will be read from the file. Otherwise, the parameters are taken from the remainder of
 * the command line.
 *
 */
public class App
{
    public static void main( String[] args )
    {
        int exitCode = 0;
        try {
            // Parse the command and get the command-line arguments.
            String[] command = StringUtils.split(args[0], '=');
            // Get the rest of the arguments.
            args = Arrays.copyOfRange(args, 1, args.length);
            // Read in the parm file if needed.
            if (command.length == 2) {
                try {
                    // Get the parameters.
                    List<String> buffer = Parms.fromFile(new File(command[1]));
                    // Add the residual.
                    buffer.addAll(Arrays.asList(args));
                    // Form them into an array.
                    args = new String[buffer.size()];
                    args = buffer.toArray(args);
                } catch (IOException e) {
                    throw new UncheckedIOException("Error reading parameter file", e);
                }
            }
            // Compute the appropriate command object.
            ICommand runObject = null;
            boolean success = true;
            switch (command[0]) {
            case "train" :
                runObject = new ClassTrainingProcessor();
                success = execute(runObject, args);
                break;
            case "rtrain" :
                runObject = new RegressionTrainingProcessor();
                success = execute(runObject, args);
                break;
            case "search" :
                runObject = new SearchProcessor();
                success = execute(runObject, args);
                break;
            case "xvalidate" :
                runObject = new CrossValidateProcessor();
                success = execute(runObject, args);
                break;
            case "validate" :
                runObject = new ValidateProcessor();
                success = execute(runObject, args);
                break;
            case "predict" :
                runObject = new PredictionProcessor();
                success = execute(runObject, args);
                break;
            case "improve" :
                runObject = new ImproveProcessor();
                success = execute(runObject, args);
                break;
            case "multirun" :
                runObject = new MultiRunProcessor();
                success = execute(runObject, args);
                break;
            case "rfTrain" :
                runObject = new RandomForestTrainProcessor();
                success = execute(runObject, args);
                break;
            case "anova" :
                runObject = new AnovaProcessor();
                success = execute(runObject, args);
                break;
            case "meanBias" :
                runObject = new MeanBiasProcessor();
                success = execute(runObject, args);
                break;
            case "pearson" :
                runObject = new PearsonProcessor();
                success = execute(runObject, args);
                break;
            case "trainCheck" :
                runObject = new TrainCheckProcessor();
                success = execute(runObject, args);
                break;
            case "roc" :
                runObject = new RocProcessor();
                success = execute(runObject, args);
                break;
            case "sortCheck" :
                runObject = new SortCheckProcessor();
                success = execute(runObject, args);
                break;
            case "accuracy" :
                runObject = new AccuracyProcessor();
                success = execute(runObject, args);
                break;
            case "--help" :
            case "-h" :
            case "help" :
                showHelp();
                break;
            default :
                throw new ParseFailureException("Invalid command code " + command[0] + ".");
            }
            if (! success) exitCode = 255;
        } catch (Exception e) {
            e.printStackTrace();
            exitCode = 255;
        }
        // Force cleanup.
        System.exit(exitCode);
    }

    /**
     * Display all the commands.
     */
    public static void showHelp() {
        System.out.println("Available commands:");
        System.out.println();
        System.out.println("multirun     use multiple trained models to make predictions");
        System.out.println("improve      train an existing model with new data to improve it");
        System.out.println("predict      use a trained model to make predictions");
        System.out.println("rtrain       train a regression model");
        System.out.println("search       train models with multiple different hyper-parameters");
        System.out.println("train        train a classification model");
        System.out.println("validate	 test a model against a model training set");
        System.out.println("xvalidate    cross-validate a model training set");
        System.out.println("anova		 display ANOVA F-measure");
    }

    /**
     * Execute a command processor.
     *
     * @param runObject		command processor to execute
     * @param args			command-line parameters
     *
     * @return TRUE if successful, else FALSE
     */
    public static boolean execute(ICommand runObject, String[] args) {
        // Execute the command.
        boolean retVal = runObject.parseCommand(args);
        if (retVal) {
            runObject.run();
        }
        return retVal;
    }

}
