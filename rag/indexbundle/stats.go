package indexbundle

import (
	"regexp"
	"sort"
	"strings"

	"github.com/go-go-golems/ragkit/rag"
)

// Signal is a named property of one chunk that a person might want to act on.
//
// Every signal states the evidence that produced it rather than only a label,
// which is why ChunkStatistics carries the measurements beside the signal list.
// A flag that a reader cannot check is a flag a reader cannot trust.
type Signal string

const (
	// SignalAtLimit means the chunk ends because the size limit ended it, not
	// because the document structure did.
	SignalAtLimit Signal = "at-limit"
	// SignalShort means the chunk is too small to support an answer on its own.
	SignalShort Signal = "short"
	// SignalNoOverlap means the chunk begins exactly where its predecessor
	// ended. This is exact, not a heuristic: the chunker applies overlap on a
	// size cut and does not apply it on a structure cut, and comparing two byte
	// ranges detects that with no threshold and no guess.
	SignalNoOverlap Signal = "no-overlap"
	// SignalHeadingCut is short and no-overlap together. It is the fault that
	// put a 110-rune chunk into a generation prompt.
	SignalHeadingCut Signal = "heading-cut"
	// SignalFurniture means the text matched a named page-structure pattern.
	// This one is a heuristic and it will be wrong sometimes, so the matched
	// pattern travels with it.
	SignalFurniture Signal = "furniture"
)

// AllSignals is every signal Measure can produce, in display order.
//
// It exists so a caller that accepts a signal name from a user can reject an
// unknown one. Filtering on a name that no signal has otherwise returns nothing
// and looks exactly like a corpus in which nothing matched, which is the
// difference between "measured zero" and "asked the wrong question".
func AllSignals() []Signal {
	return []Signal{
		SignalAtLimit, SignalShort, SignalNoOverlap, SignalHeadingCut,
		SignalFurniture,
	}
}

// ValidSignal reports whether a name is a signal.
func ValidSignal(name string) bool {
	for _, signal := range AllSignals() {
		if string(signal) == name {
			return true
		}
	}
	return false
}

// BucketRange is one inclusive rune-count bucket of the size histogram.
type BucketRange struct {
	Low  int
	High int
}

// FurniturePattern is a named text pattern that indicates page structure rather
// than document content.
//
// Name is shown beside the flag so a person can judge the match. Expression is
// matched against the chunk text.
type FurniturePattern struct {
	Name       string
	Expression *regexp.Regexp
	// MinimumMatches is how many times the expression must match before the
	// pattern applies. Zero and one both mean once.
	//
	// A threshold is needed because the strongest signal in the real bundle is
	// density, not presence. One telephone number in a nursery guide is
	// content; fifty-nine in one chunk is a directory. A presence-only matcher
	// cannot tell those apart, and lowering the expression until it only hits
	// directories would make it unreadable.
	MinimumMatches int
	// Evidence names what a match counts, for the flag line. It completes the
	// sentence "3 telephone numbers".
	Evidence string
}

// ConfirmedFurniturePatterns are the patterns checked by hand against
// ttc-866972d249d18e770631e563346a4774, with the counts they produced there.
//
// Nothing is in this set without a count. Each is ordered most specific first,
// because the first match wins and a chunk should be named by the narrowest
// pattern that describes it.
//
//	address directory   9 chunks   1 document    (>= 3 telephone numbers)
//	contact form        1 chunk    1 document
//	navigation          1 chunk    1 document
//	product data      102 chunks  74 documents
//
// product data is by far the widest and is the one to distrust: a 1200-rune
// chunk that mentions stock_status once may be mostly prose. Its flag exists to
// start a judgement, not to finish one.
func ConfirmedFurniturePatterns() []FurniturePattern {
	return []FurniturePattern{
		{
			Name:           "address directory",
			Expression:     regexp.MustCompile(`\d{3}[-.\s]\d{3}[-.\s]\d{4}`),
			MinimumMatches: 3,
			Evidence:       "telephone numbers",
		},
		{
			Name:       "contact form",
			Expression: regexp.MustCompile(`\[wpforms id=`),
			Evidence:   "form shortcodes",
		},
		{
			Name:       "navigation",
			Expression: regexp.MustCompile(`(?i)\A\s*login\s*\z`),
			Evidence:   "navigation labels",
		},
		{
			Name:       "product data",
			Expression: regexp.MustCompile(`stock_status:\s*\w`),
			Evidence:   "stock fields",
		},
	}
}

// StatisticsOptions configures the thresholds. The zero value is usable and
// applies the defaults below.
type StatisticsOptions struct {
	// Buckets defaults to DefaultBuckets. A bucket set that does not cover a
	// chunk drops it from the histogram, so Statistics reports Uncounted rather
	// than let the histogram total silently disagree with the chunk count.
	Buckets []BucketRange
	// ShortRunes is the exclusive upper bound of SignalShort. It defaults to
	// 300.
	ShortRunes int
	// AtLimitRunes is the inclusive lower bound of SignalAtLimit. Zero means
	// the chunker maximum less atLimitSlack.
	//
	// This is a bound rather than a slack because a slack cannot express
	// itself. A slack field defaults when it is zero, and zero slack -- count
	// only chunks at the exact limit -- is a thing a caller may legitimately
	// want, so the value and the "unset" marker collided. My first version had
	// that collision and it silently counted 1778 chunks where the design doc
	// measured 1779: the default never applied, the rule became "exactly 1200",
	// and one chunk of 1190 to 1199 runes fell out. A bound has no such
	// ambiguity, because zero runes is not a bound anyone wants.
	AtLimitRunes int
	// Furniture is the pattern set. It defaults to none.
	//
	// Detection is off unless a caller asks for it. The patterns are heuristics,
	// a set that is too eager labels real content as page structure, and a
	// wrong flag on a screen that exists to be trusted costs more than a missing
	// one. Pass ConfirmedFurniturePatterns to enable the patterns that were
	// checked by hand against the real bundle.
	Furniture []FurniturePattern
}

// DefaultBuckets are the buckets the design doc measured with. They are uneven
// on purpose: nine chunks in ten land in the top 10 runes, so even buckets
// would put 89.8% of the corpus in one bar and tell a reader nothing about the
// rest.
func DefaultBuckets() []BucketRange {
	return []BucketRange{
		{Low: 0, High: 99},
		{Low: 100, High: 199},
		{Low: 200, High: 399},
		{Low: 400, High: 599},
		{Low: 600, High: 799},
		{Low: 800, High: 999},
		{Low: 1000, High: 1189},
		{Low: 1190, High: 1200},
	}
}

const (
	defaultShortRunes = 300
	// atLimitSlack is how far below the chunker maximum still counts as at the
	// limit. A rune boundary can leave a chunk a few runes short of an exact
	// limit, so an exact comparison undercounts.
	atLimitSlack = 10
)

func (o StatisticsOptions) resolved(maximumRunes int) StatisticsOptions {
	if len(o.Buckets) == 0 {
		o.Buckets = DefaultBuckets()
	}
	if o.ShortRunes <= 0 {
		o.ShortRunes = defaultShortRunes
	}
	if o.AtLimitRunes <= 0 {
		o.AtLimitRunes = maximumRunes - atLimitSlack
	}
	return o
}

// ChunkStatistics is one chunk measured, with the evidence beside the verdict.
type ChunkStatistics struct {
	ChunkID    string
	DocumentID string
	Ordinal    int
	Range      rag.Range
	Runes      int
	// OverlapBytes is how many bytes this chunk shares with its predecessor in
	// the same document. It is zero for the first chunk of a document, where
	// there is nothing to overlap, and HasPredecessor distinguishes the two.
	OverlapBytes   int
	HasPredecessor bool
	Signals        []Signal
	// FurniturePattern names the pattern that matched. It is empty unless
	// SignalFurniture is present.
	FurniturePattern string
	// FurnitureMatches and FurnitureEvidence are the count and the noun behind
	// the flag, so a screen can write "59 telephone numbers" rather than only
	// "furniture". A heuristic that shows its working can be overruled; one
	// that shows a label cannot.
	FurnitureMatches  int
	FurnitureEvidence string
	// Excerpt is the first line of the chunk text, trimmed. It exists so a
	// caller listing flagged chunks can show what was flagged without holding
	// every chunk body.
	Excerpt string
}

// Has reports whether a signal is present.
func (c ChunkStatistics) Has(signal Signal) bool {
	for _, present := range c.Signals {
		if present == signal {
			return true
		}
	}
	return false
}

// Bucket is one histogram bar.
type Bucket struct {
	BucketRange
	Count int
	Share float64
}

// SignalSummary counts one signal across the bundle.
type SignalSummary struct {
	Signal Signal
	Chunks int
	// Documents is how many documents hold at least one chunk with this signal.
	// A count of chunks alone cannot say whether a fault is spread across the
	// corpus or concentrated in one document, and that difference decides what
	// a person does about it.
	Documents int
}

// Statistics is the whole bundle measured.
type Statistics struct {
	Chunks    []ChunkStatistics
	Histogram []Bucket
	// Uncounted is how many chunks fell outside every bucket. It is reported
	// rather than ignored so a histogram whose bars do not sum to the chunk
	// count says so.
	Uncounted int
	Signals   []SignalSummary
	// Percentiles are over the rune counts, keyed by percentile. A distribution
	// whose 25th and 95th percentiles are the same number is a distribution
	// with one value in it, and that is the headline fact about this chunker.
	Percentiles map[int]int
	TotalChunks int
	// MaximumRunes is the chunker limit the signals were computed against.
	MaximumRunes int
}

// Count returns the chunk count for one signal.
func (s Statistics) Count(signal Signal) int {
	for _, summary := range s.Signals {
		if summary.Signal == signal {
			return summary.Chunks
		}
	}
	return 0
}

// DocumentCount returns how many documents hold one signal.
func (s Statistics) DocumentCount(signal Signal) int {
	for _, summary := range s.Signals {
		if summary.Signal == signal {
			return summary.Documents
		}
	}
	return 0
}

// Flagged returns the chunks carrying one signal, in bundle order.
func (s Statistics) Flagged(signal Signal) []ChunkStatistics {
	var flagged []ChunkStatistics
	for _, chunk := range s.Chunks {
		if chunk.Has(signal) {
			flagged = append(flagged, chunk)
		}
	}
	return flagged
}

// Measure computes the statistics for an inspection.
//
// It is a pure function over the chunks and the chunker identity. Nothing here
// reads a file, so a caller can measure a bundle it built in memory as easily
// as one it read from disk.
func Measure(
	inspection *Inspection, options StatisticsOptions,
) Statistics {
	if inspection == nil {
		return Statistics{Histogram: emptyHistogram(options.resolved(0).Buckets)}
	}
	return measureChunks(
		inspection.Chunks, inspection.Manifest.Chunker.MaximumRunes, options,
	)
}

func measureChunks(
	chunks []rag.Chunk, maximumRunes int, options StatisticsOptions,
) Statistics {
	options = options.resolved(maximumRunes)

	// Predecessors are found per document by ordinal, not by position in the
	// slice. chunks.json is written in document order today; relying on that
	// would compare a chunk to a neighbour from a different document the moment
	// it stopped being true, and every overlap number would be wrong in a way
	// that still looked plausible.
	previousEnd := predecessorEnds(chunks)

	statistics := Statistics{
		Chunks:       make([]ChunkStatistics, 0, len(chunks)),
		TotalChunks:  len(chunks),
		MaximumRunes: maximumRunes,
	}
	runeCounts := make([]int, 0, len(chunks))
	counts := make(map[Signal]int)
	documents := make(map[Signal]map[string]struct{})
	note := func(signal Signal, documentID string) {
		counts[signal]++
		if documents[signal] == nil {
			documents[signal] = make(map[string]struct{})
		}
		documents[signal][documentID] = struct{}{}
	}

	for _, chunk := range chunks {
		runes := len([]rune(chunk.Text))
		runeCounts = append(runeCounts, runes)
		measured := ChunkStatistics{
			ChunkID: chunk.ID, DocumentID: chunk.DocumentID,
			Ordinal: chunk.Ordinal, Range: chunk.Range, Runes: runes,
			Excerpt: excerptOf(chunk.Text),
		}
		key := documentOrdinal{document: chunk.DocumentID, ordinal: chunk.Ordinal}
		if end, ok := previousEnd[key]; ok {
			measured.HasPredecessor = true
			measured.OverlapBytes = max(0, end-chunk.Range.ByteStart)
		}

		if options.AtLimitRunes > 0 && runes >= options.AtLimitRunes {
			measured.Signals = append(measured.Signals, SignalAtLimit)
			note(SignalAtLimit, chunk.DocumentID)
		}
		short := runes < options.ShortRunes
		if short {
			measured.Signals = append(measured.Signals, SignalShort)
			note(SignalShort, chunk.DocumentID)
		}
		// The first chunk of a document has no predecessor, so it cannot lack
		// overlap with one. Counting it would report every document as holding
		// the fault.
		noOverlap := measured.HasPredecessor && measured.OverlapBytes == 0
		if noOverlap {
			measured.Signals = append(measured.Signals, SignalNoOverlap)
			note(SignalNoOverlap, chunk.DocumentID)
		}
		if short && noOverlap {
			measured.Signals = append(measured.Signals, SignalHeadingCut)
			note(SignalHeadingCut, chunk.DocumentID)
		}
		if pattern, matches, ok := matchFurniture(chunk.Text, options.Furniture); ok {
			measured.Signals = append(measured.Signals, SignalFurniture)
			measured.FurniturePattern = pattern.Name
			measured.FurnitureMatches = matches
			measured.FurnitureEvidence = pattern.Evidence
			note(SignalFurniture, chunk.DocumentID)
		}
		statistics.Chunks = append(statistics.Chunks, measured)
	}

	statistics.Histogram, statistics.Uncounted = histogram(
		runeCounts, options.Buckets,
	)
	statistics.Percentiles = percentiles(runeCounts, []int{25, 50, 75, 95})
	statistics.Signals = summarise(counts, documents)
	return statistics
}

type documentOrdinal struct {
	document string
	ordinal  int
}

// predecessorEnds maps each chunk to the byte end of the chunk before it in the
// same document.
func predecessorEnds(chunks []rag.Chunk) map[documentOrdinal]int {
	byDocument := make(map[string][]rag.Chunk)
	for _, chunk := range chunks {
		byDocument[chunk.DocumentID] = append(byDocument[chunk.DocumentID], chunk)
	}
	ends := make(map[documentOrdinal]int, len(chunks))
	for document, group := range byDocument {
		sort.SliceStable(group, func(a, b int) bool {
			return group[a].Ordinal < group[b].Ordinal
		})
		for index := 1; index < len(group); index++ {
			ends[documentOrdinal{document: document, ordinal: group[index].Ordinal}] =
				group[index-1].Range.ByteEnd
		}
	}
	return ends
}

func histogram(runeCounts []int, buckets []BucketRange) ([]Bucket, int) {
	bars := emptyHistogram(buckets)
	uncounted := 0
	for _, runes := range runeCounts {
		placed := false
		for index := range bars {
			if runes >= bars[index].Low && runes <= bars[index].High {
				bars[index].Count++
				placed = true
				break
			}
		}
		if !placed {
			uncounted++
		}
	}
	if len(runeCounts) > 0 {
		for index := range bars {
			bars[index].Share =
				float64(bars[index].Count) / float64(len(runeCounts)) * 100
		}
	}
	return bars, uncounted
}

func emptyHistogram(buckets []BucketRange) []Bucket {
	bars := make([]Bucket, 0, len(buckets))
	for _, bucket := range buckets {
		bars = append(bars, Bucket{BucketRange: bucket})
	}
	return bars
}

// percentiles uses nearest-rank on a sorted copy. It is not interpolated: a
// percentile of a rune count must be a rune count that a chunk actually has,
// or the number cannot be checked against the corpus.
func percentiles(runeCounts []int, wanted []int) map[int]int {
	result := make(map[int]int, len(wanted))
	if len(runeCounts) == 0 {
		return result
	}
	sorted := make([]int, len(runeCounts))
	copy(sorted, runeCounts)
	sort.Ints(sorted)
	for _, percentile := range wanted {
		rank := percentile * len(sorted) / 100
		if rank >= len(sorted) {
			rank = len(sorted) - 1
		}
		result[percentile] = sorted[rank]
	}
	return result
}

// summarise orders the signals for display. The order is fixed rather than
// sorted by count, so the same signal is in the same place on every screen and
// on every bundle.
func summarise(
	counts map[Signal]int, documents map[Signal]map[string]struct{},
) []SignalSummary {
	order := AllSignals()
	summaries := make([]SignalSummary, 0, len(order))
	for _, signal := range order {
		summaries = append(summaries, SignalSummary{
			Signal: signal, Chunks: counts[signal],
			Documents: len(documents[signal]),
		})
	}
	return summaries
}

// matchFurniture returns the first pattern that reaches its threshold, with the
// number of matches it found.
//
// First match wins, so callers order their patterns most specific first. The
// count is taken with FindAllString rather than MatchString because a threshold
// needs the number, and a pattern with a threshold of one costs the same either
// way.
func matchFurniture(
	text string, patterns []FurniturePattern,
) (FurniturePattern, int, bool) {
	for _, pattern := range patterns {
		if pattern.Expression == nil {
			continue
		}
		threshold := max(1, pattern.MinimumMatches)
		matches := len(pattern.Expression.FindAllString(text, threshold))
		if matches >= threshold {
			return pattern, len(pattern.Expression.FindAllString(text, -1)), true
		}
	}
	return FurniturePattern{}, 0, false
}

// excerptOf is the first non-empty line, collapsed and bounded.
func excerptOf(text string) string {
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		trimmed = strings.Join(strings.Fields(trimmed), " ")
		if len([]rune(trimmed)) > 80 {
			return string([]rune(trimmed)[:80])
		}
		return trimmed
	}
	return ""
}

// SignalsByDocument returns the distinct signals of each document's chunks.
//
// The signals of each document keep the fixed display order. They do not keep
// the order in which the fold met them. Map iteration and chunk order both vary
// between runs, so an order that follows them makes the same document read
// differently each time.
//
// A document with no signals gets no entry. An empty slice and a missing key
// carry the same meaning here, and a missing key costs nothing.
func (s Statistics) SignalsByDocument() map[string][]Signal {
	present := make(map[string]map[Signal]struct{})
	for _, chunk := range s.Chunks {
		if len(chunk.Signals) == 0 {
			continue
		}
		if present[chunk.DocumentID] == nil {
			present[chunk.DocumentID] = make(map[Signal]struct{})
		}
		for _, signal := range chunk.Signals {
			present[chunk.DocumentID][signal] = struct{}{}
		}
	}
	result := make(map[string][]Signal, len(present))
	for document, signals := range present {
		for _, signal := range AllSignals() {
			if _, ok := signals[signal]; ok {
				result[document] = append(result[document], signal)
			}
		}
	}
	return result
}

// CountByDocument returns how many chunks of one document carry one signal.
//
// A flag needs a count. "3 heading cuts" tells a person how much of the
// document the fault covers. "heading-cut" does not.
func (s Statistics) CountByDocument(documentID string, signal Signal) int {
	count := 0
	for _, chunk := range s.Chunks {
		if chunk.DocumentID == documentID && chunk.Has(signal) {
			count++
		}
	}
	return count
}

// FurniturePatternsOf returns the furniture patterns that match a document, in
// first-seen order.
func (s Statistics) FurniturePatternsOf(documentID string) []string {
	var patterns []string
	seen := make(map[string]struct{})
	for _, chunk := range s.Chunks {
		if chunk.DocumentID != documentID || chunk.FurniturePattern == "" {
			continue
		}
		if _, ok := seen[chunk.FurniturePattern]; ok {
			continue
		}
		seen[chunk.FurniturePattern] = struct{}{}
		patterns = append(patterns, chunk.FurniturePattern)
	}
	return patterns
}
