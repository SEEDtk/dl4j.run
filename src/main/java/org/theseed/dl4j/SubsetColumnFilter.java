/**
 *
 */
package org.theseed.dl4j;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * This filter is designed to restrict the input columns to the first N in the specified list.
 *
 * @author Bruce Parrello
 *
 */
public class SubsetColumnFilter extends BalanceColumnFilter {

    // FIELDS
    /** set of column names to keep */
    private Set<String> goodCols;

    /**
     * Create a new subset filter.
     *
     * @param fieldList		complete list of input columns
     * @param len			number of input columns (from the beginning of the list) to keep
     * @param metaCols		list of metadata columns
     * @param labelCols 	list of label columns
     */
    public SubsetColumnFilter(List<String> fieldList, int len, List<String> metaCols, List<String> labelCols) {
        // Create a set of the columns we want to keep.
        this.goodCols = new HashSet<String>(2 * len);
        this.goodCols.addAll(metaCols);
        this.goodCols.addAll(labelCols);
        if (len >= fieldList.size())
            this.goodCols.addAll(fieldList);
        else
            this.goodCols.addAll(fieldList.subList(0, len));
    }

    @Override
    public boolean allows(String string) {
        return this.goodCols.contains(string);
    }

}
