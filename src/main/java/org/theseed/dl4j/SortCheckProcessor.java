/**
 *
 */
package org.theseed.dl4j;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.io.TabbedLineReader;
import org.theseed.utils.BasePipeProcessor;
import org.theseed.utils.ParseFailureException;

/**
 * This method seeks to find patterns in high-performing examples.  The examples are sorted from highest
 * to lowest by a specified output column.  Then all the other columns containing only 1s and 0s are processed.
 * For each output level, we output the IDs of the columns that are either mostly 0 or mostly 1, where "mostly"
 * is defined by a command-line parameter.
 *
 * The positional parameter is the name of the output column to sort.
 *
 * The command-line options are as follows:
 *
 * -h	display command-line usage
 * -v	display more frequent log messages
 * -i	name of the input xmatrix file (if not STDIN)
 * -o	name of the output file for the report (if not STDOUT)
 *
 * --min	minimum fraction of the column values that must be one thing or the other in order to qualify as "mostly";
 * 			the default is 0.95
 *
 * @author Bruce Parrello
 *
 */
public class SortCheckProcessor extends BasePipeProcessor {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(SortCheckProcessor.class);

    // COMMAND-LINE OPTIONS

    /** minimum fraction of the column values that must be 1 or 0 in order to qualify for output */
    @Option(name = "--min", metaVar = "0.80", usage = "fraction of column values that must be 1 or 0 to qualify for output")
    private double minFraction;

    /** name of the output column */
    @Argument(index = 0, metaVar = "outCol", usage = "name of the output column", required = true)
    private String outColName;

    @Override
    protected void setPipeDefaults() {
        this.minFraction = 0.95;
    }

    @Override
    protected void validatePipeInput(TabbedLineReader inputStream) throws IOException {
        if (inputStream.findColumn(outColName) < 0)
            throw new IOException("Output column name not found in input file.");
    }

    @Override
    protected void validatePipeParms() throws IOException, ParseFailureException {
        if (this.minFraction <= 0.0 || this.minFraction > 1.0)
            throw new ParseFailureException("Minimum fraction must be between 0.0 and 1.0.");
    }

    @Override
    protected void runPipeline(TabbedLineReader inputStream, PrintWriter writer) throws Exception {
        // Read the input file to get the counts.
        log.info("Reading input to get counts for each {} value.", outColName);
        var countTable = new CountRowTable(inputStream, outColName);
        // Write the output header.
        writer.println(this.outColName + "\tcount\tusually_present\tusually_absent");
        // This will track the number of records encountered so far.
        int recordCount = 0;
        // Get the names of the input columns of interest.
        String[] colNames = countTable.getColNames();
        // Set up totals for each column.
        int[] colCounts = new int[colNames.length];
        // These will hold the column names to output for each value.
        var mostly1 = new ArrayList<String>(colNames.length);
        var mostly0 = new ArrayList<String>(colNames.length);
        // Now loop through the count rows.
        List<CountRow> counts = countTable.sortedRows();
        for (CountRow count : counts) {
            mostly1.clear();
            mostly0.clear();
            // Compute the count threshold for this output value.
            recordCount += count.getRecordCount();
            int mostly1Min = (int) Math.ceil(this.minFraction * recordCount);
            int mostly0Max = recordCount - mostly1Min;
            // Now run through the columns, accumulating the mostly-1 and mostly-0 column names.
            count.accumulate(colCounts);
            for (int i = 0; i < colCounts.length; i++) {
                if (colCounts[i] >= mostly1Min)
                    mostly1.add(colNames[i]);
                else if (colCounts[i] <= mostly0Max)
                    mostly0.add(colNames[i]);
            }
            // Write this output line.
            writer.format("%6.4f\t%d\t%s\t%s%n", count.getOutVal(), recordCount, StringUtils.join(mostly1, ", "),
                    StringUtils.join(mostly0, ", "));
        }
    }

}
