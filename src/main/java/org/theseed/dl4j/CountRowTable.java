/**
 *
 */
package org.theseed.dl4j;

import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.io.TabbedLineReader;

/**
 * This is a map of output values to count-row objects.  It is loaded from a tab-delimited x-matrix type file,
 * and defined by the output column name.
 *
 * During the load, we identify the the input columns that contain only 1s and 0s.  Each row is stored as a value
 * and a bitset defining the columns with 1s.  Another bitset defines the columns that only have 0s and 1s.
 * After the load, we convert this into the map of count-rows.
 *
 * @author Bruce Parrello
 *
 */
public class CountRowTable {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(CountRowTable.class);
    /** map of output values to count rows */
    private Map<Double, CountRow> countMap;
    /** array of column names for columns kept */
    private String[] colNames;

    /**
     * This class holds the initial row information before we create the count objects.
     */
    private class RowDatum {

        /** output value */
        private double outValue;
        /** set of columns with 1s in them */
        private BitSet activeCols;

        /**
         * Create a new row datum.
         *
         * @param outVal
         * @param row
         */
        public RowDatum(double outVal, BitSet row) {
            this.outValue = outVal;
            this.activeCols = row;
        }

        /**
         * @return the outValue
         */
        protected double getOutValue() {
            return this.outValue;
        }

        /**
         * @return the activeCols
         */
        protected BitSet getActiveCols() {
            return this.activeCols;
        }

    }
    /**
     * Load a count-row map from an input stream.
     *
     * @param input		open TabbedLineReader containing the input
     * @param outCol	name of the output column
     *
     * @throws IOException
     */
    public CountRowTable(TabbedLineReader input, String outCol) throws IOException {
        // Create a list to hold the initial row data.
        var rows = new ArrayList<RowDatum>(100);
        // Get the number of columns and the index of the output column.
        int nCols = input.size();
        int outColIdx = input.findField(outCol);
        // This bit set will identify the columns that are still good.
        BitSet goodCols = new BitSet(nCols);
        goodCols.set(0, nCols);
        goodCols.clear(outColIdx);
        // Now start reading records.
        for (TabbedLineReader.Line line : input) {
            // Get the output value.
            double outVal = line.getDouble(outColIdx);
            // Create a bit set for this line.
            BitSet row = new BitSet(nCols);
            // Scan the columns.
            for (int i = 0; i < nCols; i++) {
                if (goodCols.get(i)) {
                    try {
                        double inVal = line.getDouble(i);
                        if (inVal == 1.0)
                            row.set(i);
                        else if (inVal != 0.0)
                            goodCols.clear(i);
                    } catch (Exception e) {
                        goodCols.clear(i);
                    }
                }
            }
            rows.add(new RowDatum(outVal, row));
        }
        // Now we have all the records in memory and we know which columns are valid.
        // Save the column labels we are keeping.
        String[] labels = input.getLabels();
        this.colNames = goodCols.stream().mapToObj(i -> labels[i]).toArray(String[]::new);
        log.info("{} records read.  {} input columns identified.", rows.size(), this.colNames.length);
        // Now build the count rows.
        this.countMap = new HashMap<Double, CountRow>(rows.size() * 4 / 3);
        for (RowDatum row : rows) {
            double outVal = row.getOutValue();
            CountRow rowCounts = this.countMap.computeIfAbsent(outVal, x -> new CountRow(x, this.colNames.length));
            rowCounts.count(goodCols, row.getActiveCols());
        }
        log.info("{} output values indentified.", this.countMap.size());
    }
    /**
     * @return a list of the count-row objects, sorted by the output value in descending order
     */
    public List<CountRow> sortedRows() {
        var retVal = new ArrayList<CountRow>(this.countMap.values());
        Collections.sort(retVal);
        return retVal;
    }
    /**
     * @return the array of column names for the columns used
     */
    public String[] getColNames() {
        return this.colNames;
    }

}
