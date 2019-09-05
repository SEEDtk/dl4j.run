package org.theseed.dl4j;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.theseed.dl4j.predict.PredictionProcessor;
import org.theseed.dl4j.train.ImproveProcessor;
import org.theseed.dl4j.train.SearchProcessor;
import org.theseed.dl4j.train.TrainingProcessor;
import org.theseed.utils.ICommand;
import org.theseed.utils.Parms;

/**
 * Main entry point for the Deep Learning utility.  The first parameter is a command-- use "train" to
 * train a model with a training set.  Use "predict" to apply a model to a prediction set.  Use
 * "search" to test multiple model configurations.
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
                runObject = new TrainingProcessor();
                success = execute(runObject, args);
                break;
            case "predict" :
                runObject = new PredictionProcessor();
                success = execute(runObject, args);
                break;
            case "search" :
                runObject = new SearchProcessor();
                success = execute(runObject, args);
                break;
            case "improve" :
                runObject = new ImproveProcessor();
                success = execute(runObject, args);
                break;
            case "--help" :
            case "-h" :
            case "help" :
                args = new String[] { "--help" };
                System.err.println("Command code \"train\":");
                runObject = new TrainingProcessor();
                execute(runObject, args);
                System.err.println();
                System.err.println("Command code \"predict\":");
                runObject = new PredictionProcessor();
                execute(runObject, args);
                System.err.println();
                System.err.println("Command code \"search\":");
                runObject = new SearchProcessor();
                execute(runObject, args);
                System.err.println();
                System.err.println("Command code \"improve\":");
                runObject = new ImproveProcessor();
                execute(runObject, args);
                break;
            default :
                throw new IllegalArgumentException("Invalid command code " + command[0] + ".");
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
