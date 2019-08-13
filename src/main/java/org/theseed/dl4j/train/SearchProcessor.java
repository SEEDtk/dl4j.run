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
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.theseed.dl4j.App;
import org.theseed.utils.ICommand;
import org.theseed.utils.Parms;

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
 * @author Bruce Parrello
 *
 */
public class SearchProcessor implements ICommand {

    // FIELDS
    /** iterator that runs through the parameter combinations */
    private Parms parmIterator;

    // COMMAND LINE

    /** help option */
    @Option(name="-h", aliases={"--help"}, help=true)
    private boolean help;

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
            parser.parseArgument(args);
            if (this.help) {
                parser.printUsage(System.err);
            } else {
                // Verify the model directory and read the parm file.
                if (! this.modelDir.isDirectory()) {
                    throw new FileNotFoundException("Model directory " + this.modelDir + " not found or invalid.");
                } else {
                    this.parmIterator = new Parms(this.parmFile);
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
        TrainingProcessor processor = new TrainingProcessor();
        // These variables track our progress and success.
        int iteration = 1;
        double bestAccuracy = 0.0;
        int bestIteration = 0;
        // Set up our summary matrix.  Note each array will contain one entry per parameter plus a slot for accuracy.
        HashMap<String, String> varMap = this.parmIterator.getVariables();
        String[] headings = ArrayUtils.insert(varMap.size(), this.parmIterator.getOptions(), "    Accuracy");
        ArrayList<String[]> data = new ArrayList<String[]>();
        data.add(headings);
        // This is a buffer to hold the parameters.
        String[] parmBuffer = new String[this.parmIterator.size() + 3];
        while (this.parmIterator.hasNext()) {
            // Create the parameter array.
            List<String> theseParms = this.parmIterator.next();
            // Save the values.
            String[] values = new String[varMap.size() + 1];
            for (int i = 0; i < headings.length; i++)
                values[i] = varMap.get(headings[i]);
            // Add the comment and the model directory.
            theseParms.add("--comment");
            theseParms.add(String.format("Iteration %d: %s", iteration, this.parmIterator));
            theseParms.add(this.modelDir.getPath());
            // Run the experiment.
            String[] actualParms = theseParms.toArray(parmBuffer);
            App.execute(processor, actualParms);
            // Save the accuracy.
            double newAccuracy = processor.getAccuracy();
            values[varMap.size()] = String.format("%12.4f", newAccuracy);
            // Compare the accuracy.
            if (newAccuracy > bestAccuracy) {
                bestIteration = iteration;
                bestAccuracy = newAccuracy;
            }
            // Save this row of the summary array.
            data.add(values);
            // Count the iteration.
            iteration++;
        }
        TrainingProcessor.log.info("** Best iteration was {} with accuracy {}.", bestIteration, bestAccuracy);
        // Now display the result matrix.  First we compute the width for each column.
        int[] widths = new int[varMap.size() + 1];
        Arrays.fill(widths, 8);
        for (String[] cols : data)
            for (int i = 0; i < widths.length; i++)
                widths[i] = Math.max(cols[i].length(), widths[i]);
        // Compute the total width.  We have 6 at the beginning and an extra space before each column.
        int totWidth = Arrays.stream(widths).sum() + widths.length + 6;
        String boundary = StringUtils.repeat('=', totWidth);
        TrainingProcessor.log.info(boundary);
        // Write out the heading line.
        TrainingProcessor.log.info(this.writeLine(widths, totWidth, "#", data.get(0)));
        // Write out a space.
        TrainingProcessor.log.info("");
        // Write out the data lines.
        for (int i = 1; i < data.size(); i++)
            TrainingProcessor.log.info(this.writeLine(widths, totWidth, Integer.toString(i), data.get(i)));
        // Write out a trailer line.
        TrainingProcessor.log.info(boundary);
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
        retVal.append(StringUtils.leftPad(idx, 6));
        // Loop through the data columns.
        for (int i = 0; i < widths.length; i++) {
            retVal.append(' ');
            retVal.append(StringUtils.center(cols[i], widths[i]));
        }
        return retVal.toString();
    }

}
