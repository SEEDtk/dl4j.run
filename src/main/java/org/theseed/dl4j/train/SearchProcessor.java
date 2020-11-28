/**
 *
 */
package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.text.TextStringBuilder;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.App;
import org.theseed.utils.ICommand;
import org.theseed.utils.MultiParms;

/**
 * In search mode, we accept a parameter file that specifies multiple values for one or more
 * parameters.  We run the training processor on each value combination.  This should be done
 * with a small number of iterations to keep performance rational.
 *
 * There are two positional parameters-- the name of the parameter file and the name of the
 * model directory.  The parameters in the parameter file are defined by the TrainingProcessor
 * class, however, no value should be specified for the "--comment" option, and the model
 * directory name will come from the command line, not the parameter file.
 *
 * The command-line options are as follows.
 *
 * -h	display command-line usage
 * -t	model type (REGRESSION or CLASS)
 *
 * --saveAll	if specified, each model will be saved to a different file, with the name "modelXX.ser", where
 * 				XX is the iteration number
 *
 * @author Bruce Parrello
 *
 */
public class SearchProcessor implements ICommand {

    /** logging facility */
    private static Logger log = LoggerFactory.getLogger(ClassTrainingProcessor.class);

    // FIELDS
    /** iterator that runs through the parameter combinations */
    private MultiParms parmIterator;

    // COMMAND LINE

    /** help option */
    @Option(name="-h", aliases={"--help"}, help=true)
    private boolean help;

    /** type of model */
    @Option(name="--type", aliases={"-t"}, usage="type of model")
    private TrainingProcessor.Type modelType;

    /** if specified, all models will be saved */
    @Option(name = "--saveAll", usage = "save all models")
    private boolean saveAll;

    /** parameter file */
    @Argument(index=0, metaVar="parms.prm", usage="parameter file with tab-separated alternatives", required=true)
    private File parmFile;

    /** model directory */
    @Argument(index=1, metaVar="modelDir", usage="model directory", required=true)
    private File modelDir;


    @Override
    public boolean parseCommand(String[] args) {
        boolean retVal = false;
        // Parse the command line.
        CmdLineParser parser = new CmdLineParser(this);
        try {
            this.help = false;
            this.saveAll = false;
            this.modelType = TrainingProcessor.Type.CLASS;
            parser.parseArgument(args);
            if (this.help) {
                parser.printUsage(System.err);
            } else {
                // Verify the model directory and read the parm file.
                if (! this.modelDir.isDirectory()) {
                    throw new FileNotFoundException("Model directory " + this.modelDir + " not found or invalid.");
                } else {
                    this.parmIterator = new MultiParms(this.parmFile);
                    // Denote we have been successful.
                    retVal = true;
                }
            }
        } catch (CmdLineException e) {
            System.err.println(e.getMessage());
            // For parameter errors, we display the command usage.
            parser.printUsage(System.err);
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
        return retVal;
    }

    @Override
    public void run() {
        // We loop through the parameter combinations, calling the training processor.
        TrainingProcessor processor = TrainingProcessor.create(modelType);
        // Suppress saving of the model unless we force it.
        processor.setSearchMode();
        // These variables track our progress and success.
        int iteration = 1;
        double bestRating = Double.NEGATIVE_INFINITY;
        int bestIteration = 0;
        // Set up our summary matrix.  Note each array will contain one entry per parameter plus a slot for accuracy.
        HashMap<String, String> varMap = this.parmIterator.getVariables();
        String[] headings = ArrayUtils.insert(varMap.size(), this.parmIterator.getOptions(), "    Rating");
        ArrayList<String[]> data = new ArrayList<String[]>();
        data.add(headings);
        while (this.parmIterator.hasNext()) {
            // If we are saving all models, we must add or replace the model name.
            if (this.saveAll) {
                File modelFile = new File(this.modelDir, String.format("model%d.ser", iteration));
                this.parmIterator.replace("--name", modelFile.toString());
            }
            // Update the comment and create the parameter array.
            this.parmIterator.replace("--comment", String.format("Iteration %d: %s", iteration, this.parmIterator));
            List<String> theseParms = this.parmIterator.next();
            // Save the varying values.
            String[] values = new String[varMap.size() + 1];
            for (int i = 0; i < headings.length; i++)
                values[i] = varMap.get(headings[i]);
            // Add the model directory to the parameters.
            theseParms.add(this.modelDir.getPath());
            // Run the experiment.
            // This is a buffer to hold the parameters.
            String[] parmBuffer = new String[this.parmIterator.size()];
            String[] actualParms = theseParms.toArray(parmBuffer);
            try {
                App.execute(processor, actualParms);
                // Save the accuracy.
                double newRating = processor.getRating();
                values[varMap.size()] = String.format("%12.4f", newRating);
                // Compare the rating.
                boolean save = this.saveAll;
                if (newRating > bestRating) {
                    // Here this is our best model.  Remember that and save the model to disk.
                    bestIteration = iteration;
                    bestRating = newRating;
                    save = true;
                    log.info("** Best iteration so far.");
                } else {
                    log.info("** Best iteration was {} with rating {}.", bestIteration, bestRating);
                }
                if (save)
                    processor.saveModelForced();
                // Save this row of the summary array.
                data.add(values);
            } catch (Exception e) {
                log.error("Exception in iteration {}: {}", iteration, e.getMessage());
                e.printStackTrace(System.err);
                log.error("Iteration aborted due to error.");
            }
            // Count the iteration.
            iteration++;
        }
        // Now display the result matrix.  First we compute the width for each column.
        int[] widths = new int[varMap.size() + 1];
        Arrays.fill(widths, 8);
        for (String[] cols : data)
            for (int i = 0; i < widths.length; i++)
                widths[i] = Math.max(cols[i].length(), widths[i]);
        // Compute the total width.  We have 7 at the beginning and an extra space before each column.
        int totWidth = Arrays.stream(widths).sum() + widths.length + 7;
        String boundary = StringUtils.repeat('=', totWidth);
        // We will build the report in here.
        TextStringBuilder buffer = new TextStringBuilder((totWidth + 2) * (data.size() + 4));
        buffer.appendNewLine();
        buffer.appendln(boundary);
        // Write out the heading line.
        buffer.appendln(this.writeLine(widths, totWidth, "# ", data.get(0)));
        // Write out a space.
        buffer.appendln("");
        // Write out the data lines.
        for (int i = 1; i < data.size(); i++) {
            String label = Integer.toString(i) + (i == bestIteration ? "*" : " ");
            buffer.appendln(this.writeLine(widths, totWidth, label, data.get(i)));
        }
        // Write out a trailer line.
        buffer.appendln(boundary);
        // Write it to the output log.
        String report = buffer.toString();
        log.info(report);
        // Write it to the trial file.
        try {
            RunStats.writeTrialReport(processor.getTrialFile(), "Summary of Search-Mode Results", report);
        } catch (IOException e) {
            System.err.println("Error writing trial.log:" + e.getMessage());
        }
    }


    /**
     * Format an output line from the string array. All the values are centered in the specified width,
     * with an index column in front (width 6) and spaces between all the columns.
     *
     * @param widths	array of column widths
     * @param totWidth	the total expected line width
     * @param idx		index label for the current row
     * @param cols		array of column values
     *
     * @return a string ready to be written.
     */
    private String writeLine(int[] widths, int totWidth, String idx, String[] cols) {
        StringBuilder retVal = new StringBuilder(totWidth);
        // Start with the line label.
        retVal.append(StringUtils.leftPad(idx, 7));
        // Loop through the data columns.
        for (int i = 0; i < widths.length; i++) {
            retVal.append(' ');
            retVal.append(StringUtils.center(cols[i], widths[i]));
        }
        return retVal.toString();
    }

}
