package indexbundle

import (
	"os"
	"regexp"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

// no-overlap compares two byte ranges exactly. It uses no threshold, so this
// test uses ranges that differ by one byte in each direction.
func TestNoOverlapIsExactAndNeedsAPredecessor(t *testing.T) {
	chunks := []rag.Chunk{
		// Ordinal 0 has no predecessor. It cannot lack overlap with one.
		{ID: "c0", DocumentID: "d", Ordinal: 0,
			Range: rag.Range{ByteStart: 0, ByteEnd: 100}, Text: "0"},
		// Begins exactly where c0 ended: no overlap.
		{ID: "c1", DocumentID: "d", Ordinal: 1,
			Range: rag.Range{ByteStart: 100, ByteEnd: 200}, Text: "1"},
		// One byte of overlap is overlap.
		{ID: "c2", DocumentID: "d", Ordinal: 2,
			Range: rag.Range{ByteStart: 199, ByteEnd: 300}, Text: "2"},
		// A gap is not overlap either, and it is not negative overlap.
		{ID: "c3", DocumentID: "d", Ordinal: 3,
			Range: rag.Range{ByteStart: 301, ByteEnd: 400}, Text: "3"},
	}
	statistics := measureChunks(chunks, 1200, StatisticsOptions{})

	byID := map[string]ChunkStatistics{}
	for _, chunk := range statistics.Chunks {
		byID[chunk.ChunkID] = chunk
	}

	require.False(t, byID["c0"].HasPredecessor)
	require.False(
		t, byID["c0"].Has(SignalNoOverlap),
		"the first chunk of a document must not carry the fault",
	)

	require.True(t, byID["c1"].HasPredecessor)
	require.Equal(t, 0, byID["c1"].OverlapBytes)
	require.True(t, byID["c1"].Has(SignalNoOverlap))

	require.Equal(t, 1, byID["c2"].OverlapBytes)
	require.False(
		t, byID["c2"].Has(SignalNoOverlap),
		"one byte of overlap is overlap",
	)

	require.Equal(
		t, 0, byID["c3"].OverlapBytes,
		"a gap gives zero overlap, never a negative count",
	)
	require.True(t, byID["c3"].Has(SignalNoOverlap))

	require.Equal(t, 2, statistics.Count(SignalNoOverlap))
	require.Equal(t, 1, statistics.DocumentCount(SignalNoOverlap))
}

// A predecessor is the previous ordinal in the same document, not the previous
// entry in the slice. Relying on file order would compare a chunk to a
// neighbour from another document and produce a wrong number that still looks
// plausible.
func TestPredecessorsAreFoundPerDocumentNotByFileOrder(t *testing.T) {
	interleaved := []rag.Chunk{
		{ID: "a1", DocumentID: "a", Ordinal: 1,
			Range: rag.Range{ByteStart: 80, ByteEnd: 200}, Text: "a1"},
		{ID: "b0", DocumentID: "b", Ordinal: 0,
			Range: rag.Range{ByteStart: 0, ByteEnd: 100}, Text: "b0"},
		{ID: "a0", DocumentID: "a", Ordinal: 0,
			Range: rag.Range{ByteStart: 0, ByteEnd: 100}, Text: "a0"},
		{ID: "b1", DocumentID: "b", Ordinal: 1,
			Range: rag.Range{ByteStart: 100, ByteEnd: 200}, Text: "b1"},
	}
	statistics := measureChunks(interleaved, 1200, StatisticsOptions{})
	byID := map[string]ChunkStatistics{}
	for _, chunk := range statistics.Chunks {
		byID[chunk.ChunkID] = chunk
	}

	// a1 overlaps a0 by 20 bytes even though b0 sits between them in the file.
	require.Equal(t, 20, byID["a1"].OverlapBytes)
	require.False(t, byID["a1"].Has(SignalNoOverlap))
	// b1 begins where b0 ended.
	require.Equal(t, 0, byID["b1"].OverlapBytes)
	require.True(t, byID["b1"].Has(SignalNoOverlap))
	// Neither ordinal 0 has a predecessor.
	require.False(t, byID["a0"].HasPredecessor)
	require.False(t, byID["b0"].HasPredecessor)
}

// at-limit allows slack below the chunker maximum, because a rune boundary can
// leave a chunk a few runes short of an exact limit. An exact comparison
// undercounts, which is how the first version of this code reported 1778
// where the corpus holds 1779.
func TestAtLimitAllowsSlackBelowTheChunkerMaximum(t *testing.T) {
	chunks := []rag.Chunk{
		{ID: "exact", DocumentID: "d", Ordinal: 0, Text: text(1200)},
		{ID: "slack", DocumentID: "d", Ordinal: 1, Text: text(1195)},
		{ID: "edge", DocumentID: "d", Ordinal: 2, Text: text(1190)},
		{ID: "below", DocumentID: "d", Ordinal: 3, Text: text(1189)},
	}
	statistics := measureChunks(chunks, 1200, StatisticsOptions{})
	require.Equal(
		t, 3, statistics.Count(SignalAtLimit),
		"1200, 1195 and 1190 are at the limit; 1189 is not",
	)

	// An explicit bound overrides the derived one, and zero is not that bound:
	// zero means unset, which is why this is a bound rather than a slack.
	exact := measureChunks(chunks, 1200, StatisticsOptions{AtLimitRunes: 1200})
	require.Equal(t, 1, exact.Count(SignalAtLimit))
}

// heading-cut is short and no-overlap together, and it must be both.
func TestHeadingCutRequiresBothConditions(t *testing.T) {
	chunks := []rag.Chunk{
		{ID: "c0", DocumentID: "d", Ordinal: 0,
			Range: rag.Range{ByteStart: 0, ByteEnd: 1200}, Text: text(1200)},
		// Short, and it does overlap: not a heading cut.
		{ID: "shortOverlapping", DocumentID: "d", Ordinal: 1,
			Range: rag.Range{ByteStart: 1100, ByteEnd: 1200}, Text: text(90)},
		// No overlap, and it is long: not a heading cut.
		{ID: "longNoOverlap", DocumentID: "d", Ordinal: 2,
			Range: rag.Range{ByteStart: 1200, ByteEnd: 2500}, Text: text(1200)},
		// Both.
		{ID: "both", DocumentID: "d", Ordinal: 3,
			Range: rag.Range{ByteStart: 2500, ByteEnd: 2590}, Text: text(90)},
	}
	statistics := measureChunks(chunks, 1200, StatisticsOptions{})
	byID := map[string]ChunkStatistics{}
	for _, chunk := range statistics.Chunks {
		byID[chunk.ChunkID] = chunk
	}
	require.False(t, byID["shortOverlapping"].Has(SignalHeadingCut))
	require.True(t, byID["shortOverlapping"].Has(SignalShort))
	require.False(t, byID["longNoOverlap"].Has(SignalHeadingCut))
	require.True(t, byID["longNoOverlap"].Has(SignalNoOverlap))
	require.True(t, byID["both"].Has(SignalHeadingCut))
	require.Equal(t, 1, statistics.Count(SignalHeadingCut))
}

// A histogram whose bars do not sum to the chunk count must say so. Silently
// dropping a chunk makes every share wrong by an amount nobody can see.
func TestHistogramReportsWhatItCouldNotPlace(t *testing.T) {
	chunks := []rag.Chunk{
		{ID: "in", DocumentID: "d", Ordinal: 0, Text: text(50)},
		{ID: "out", DocumentID: "d", Ordinal: 1, Text: text(5000)},
	}
	statistics := measureChunks(chunks, 1200, StatisticsOptions{
		Buckets: []BucketRange{{Low: 0, High: 99}},
	})
	require.Len(t, statistics.Histogram, 1)
	require.Equal(t, 1, statistics.Histogram[0].Count)
	require.Equal(t, 1, statistics.Uncounted)
	require.Equal(t, 2, statistics.TotalChunks)

	// The share is of every chunk measured, not of the ones that fitted, so a
	// bar reading 50% with one uncounted chunk is telling the truth.
	require.InDelta(t, 50.0, statistics.Histogram[0].Share, 0.01)
}

// Furniture detection is off unless a caller asks for it.
func TestFurnitureDetectionIsOffByDefault(t *testing.T) {
	chunks := []rag.Chunk{
		{ID: "login", DocumentID: "d", Ordinal: 0, Text: "Login"},
	}
	quiet := measureChunks(chunks, 1200, StatisticsOptions{})
	require.Equal(t, 0, quiet.Count(SignalFurniture))

	loud := measureChunks(chunks, 1200, StatisticsOptions{
		Furniture: ConfirmedFurniturePatterns(),
	})
	require.Equal(t, 1, loud.Count(SignalFurniture))
	require.Equal(t, "navigation", loud.Chunks[0].FurniturePattern)
}

// A threshold pattern must not fire below its threshold, and the flag must
// carry the count that made it fire.
func TestFurnitureThresholdStatesItsEvidence(t *testing.T) {
	patterns := []FurniturePattern{{
		Name:           "address directory",
		Expression:     regexp.MustCompile(`\d{3}-\d{3}-\d{4}`),
		MinimumMatches: 3,
		Evidence:       "telephone numbers",
	}}
	chunks := []rag.Chunk{
		{ID: "two", DocumentID: "d", Ordinal: 0,
			Text: "call 501-643-2622 or 888-868-9960 for advice"},
		{ID: "four", DocumentID: "d", Ordinal: 1,
			Text: "501-643-2622 888-868-9960 479-894-1234 870-555-0000"},
	}
	statistics := measureChunks(chunks, 1200, StatisticsOptions{Furniture: patterns})
	byID := map[string]ChunkStatistics{}
	for _, chunk := range statistics.Chunks {
		byID[chunk.ChunkID] = chunk
	}
	require.False(
		t, byID["two"].Has(SignalFurniture),
		"two telephone numbers in prose is content, not a directory",
	)
	require.True(t, byID["four"].Has(SignalFurniture))

	// The count is every match, not the threshold. A screen writes
	// "4 telephone numbers", and stopping at 3 would understate it.
	require.Equal(t, 4, byID["four"].FurnitureMatches)
	require.Equal(t, "telephone numbers", byID["four"].FurnitureEvidence)
}

// The first pattern wins, so a caller's order decides the name. This is why
// ConfirmedFurniturePatterns is ordered most specific first.
func TestFurnitureTakesTheFirstMatchingPattern(t *testing.T) {
	both := "stock_status: instock and 501-643-2622 501-643-2623 501-643-2624"
	chunks := []rag.Chunk{{ID: "c", DocumentID: "d", Ordinal: 0, Text: both}}

	confirmed := measureChunks(chunks, 1200, StatisticsOptions{
		Furniture: ConfirmedFurniturePatterns(),
	})
	require.Equal(t, "address directory", confirmed.Chunks[0].FurniturePattern)

	reversed := ConfirmedFurniturePatterns()
	for low, high := 0, len(reversed)-1; low < high; low, high = low+1, high-1 {
		reversed[low], reversed[high] = reversed[high], reversed[low]
	}
	flipped := measureChunks(chunks, 1200, StatisticsOptions{Furniture: reversed})
	require.Equal(t, "product data", flipped.Chunks[0].FurniturePattern)
}

// Percentiles are nearest-rank, not interpolated: a percentile of a rune count
// must be a count some chunk actually has, or it cannot be checked against the
// corpus.
func TestPercentilesReportAValueSomeChunkHas(t *testing.T) {
	chunks := make([]rag.Chunk, 0, 10)
	for index := range 10 {
		chunks = append(chunks, rag.Chunk{
			ID: "c", DocumentID: "d", Ordinal: index,
			Text: text(100 * (index + 1)),
		})
	}
	statistics := measureChunks(chunks, 1200, StatisticsOptions{})
	for _, percentile := range []int{25, 50, 75, 95} {
		value := statistics.Percentiles[percentile]
		require.Zero(
			t, value%100,
			"p%d = %d is not a value any chunk has", percentile, value,
		)
	}
	require.Equal(t, 300, statistics.Percentiles[25])
	require.Equal(t, 500, statistics.Percentiles[50])
	require.Equal(t, 1000, statistics.Percentiles[95])
}

// Measure tolerates a nil inspection and still returns a usable histogram, so a
// screen drawn before a bundle loads has something to draw.
func TestMeasureOfNothingIsStillAHistogram(t *testing.T) {
	statistics := Measure(nil, StatisticsOptions{})
	require.Len(t, statistics.Histogram, len(DefaultBuckets()))
	require.Equal(t, 0, statistics.TotalChunks)
	require.Empty(t, statistics.Percentiles)
}

// TestMeasuresTheRealBundle asserts the exact counts measured on
// ttc-866972d249d18e770631e563346a4774.
//
// These are equalities, not bounds. A rebuild that changes any of them must
// fail here so a person looks at why, which is the only reason to write the
// numbers down at all.
//
// Opt in with RAG_TTC_INDEX_BUNDLE=1.
func TestMeasuresTheRealBundle(t *testing.T) {
	if os.Getenv("RAG_TTC_INDEX_BUNDLE") != "1" {
		t.Skip("set RAG_TTC_INDEX_BUNDLE=1 to read the real index bundle")
	}
	t.Chdir(repoRoot(t))
	inspection, err := Inspect(t.Context(), realBundlePath)
	require.NoError(t, err)
	statistics := Measure(inspection, StatisticsOptions{
		Furniture: ConfirmedFurniturePatterns(),
	})

	require.Equal(t, 1982, statistics.TotalChunks)
	require.Equal(
		t, 0, statistics.Uncounted,
		"the default buckets must cover every chunk in this bundle",
	)

	// The size distribution. Nine chunks in ten end because the size limit
	// ended them.
	require.Equal(t, []int{4, 18, 47, 40, 40, 36, 18, 1779}, bucketCounts(statistics))
	require.InDelta(t, 89.8, statistics.Histogram[7].Share, 0.05)

	// Every percentile is the maximum. A distribution whose 25th and 95th
	// percentiles are the same number has one value in it, and that is the
	// headline fact about this chunker.
	for _, percentile := range []int{25, 50, 75, 95} {
		require.Equal(
			t, 1200, statistics.Percentiles[percentile],
			"p%d", percentile,
		)
	}

	// The signals, chunks and documents both. The document count is what says
	// whether a fault is spread or concentrated.
	require.Equal(t, 1779, statistics.Count(SignalAtLimit))
	require.Equal(t, 161, statistics.DocumentCount(SignalAtLimit))
	require.Equal(t, 45, statistics.Count(SignalShort))
	require.Equal(t, 42, statistics.DocumentCount(SignalShort))
	require.Equal(t, 4, statistics.Count(SignalNoOverlap))
	require.Equal(
		t, 1, statistics.DocumentCount(SignalNoOverlap),
		"all four no-overlap chunks are in one document",
	)
	require.Equal(t, 3, statistics.Count(SignalHeadingCut))
	require.Equal(t, 1, statistics.DocumentCount(SignalHeadingCut))

	// Three chunks in 1982, one document in 200. A tool that finds three chunks
	// in two thousand is the point of this work, and overstating how common the
	// fault is would undo that.
	require.Equal(t, 113, statistics.Count(SignalFurniture))
	require.Equal(t, 77, statistics.DocumentCount(SignalFurniture))
	require.Equal(t, map[string]int{
		"address directory": 9,
		"contact form":      1,
		"navigation":        1,
		"product data":      102,
	}, furnitureByPattern(statistics))

	// The heading cuts are ordinals 2, 3 and 4 of one document, and the chunk
	// that reached a generation prompt is the last of them.
	cuts := statistics.Flagged(SignalHeadingCut)
	require.Len(t, cuts, 3)
	require.Equal(t, []int{2, 3, 4}, []int{
		cuts[0].Ordinal, cuts[1].Ordinal, cuts[2].Ordinal,
	})
	require.Equal(t, []int{81, 106, 110}, []int{
		cuts[0].Runes, cuts[1].Runes, cuts[2].Runes,
	})
	require.Equal(t, "chunk-dfde14a88609c4b2", cuts[2].ChunkID)
	require.Equal(t, 0, cuts[2].OverlapBytes)
	require.True(t, cuts[2].HasPredecessor)

	document, ok := inspection.DocumentByID(cuts[2].DocumentID)
	require.True(t, ok)
	require.Equal(t, "Arkansas Trees For Sale", document.Title)

	// The fourth no-overlap chunk is ordinal 5, and it is not a heading cut
	// because it reaches the size limit. It is the cut that ends the run.
	overlaps := statistics.Flagged(SignalNoOverlap)
	require.Len(t, overlaps, 4)
	require.Equal(t, 5, overlaps[3].Ordinal)
	require.Equal(t, 1200, overlaps[3].Runes)
	require.False(t, overlaps[3].Has(SignalHeadingCut))

	// The address directory is ordinals 10 to 18 of the same document: nine
	// chunks holding a nursery telephone list.
	var directory []ChunkStatistics
	for _, chunk := range statistics.Flagged(SignalFurniture) {
		if chunk.FurniturePattern == "address directory" {
			directory = append(directory, chunk)
		}
	}
	require.Len(t, directory, 9)
	require.Equal(t, document.ID, directory[0].DocumentID)
	require.Equal(t, 10, directory[0].Ordinal)
	require.Equal(t, 18, directory[8].Ordinal)
	require.Equal(
		t, 59, directory[7].FurnitureMatches,
		"ordinal 17 holds 59 telephone numbers",
	)

	// The five-rune navigation chunk that reached two generation prompts and
	// was cited in neither.
	navigation := ""
	for _, chunk := range statistics.Flagged(SignalFurniture) {
		if chunk.FurniturePattern == "navigation" {
			navigation = chunk.ChunkID
			require.Equal(t, 5, chunk.Runes)
		}
	}
	require.Equal(t, "chunk-77e10898929f8bd4", navigation)
}

func bucketCounts(statistics Statistics) []int {
	counts := make([]int, 0, len(statistics.Histogram))
	for _, bucket := range statistics.Histogram {
		counts = append(counts, bucket.Count)
	}
	return counts
}

func furnitureByPattern(statistics Statistics) map[string]int {
	counts := make(map[string]int)
	for _, chunk := range statistics.Flagged(SignalFurniture) {
		counts[chunk.FurniturePattern]++
	}
	return counts
}

// text returns a string of exactly n runes.
func text(runes int) string {
	out := make([]rune, runes)
	for index := range out {
		out[index] = 'x'
	}
	return string(out)
}
