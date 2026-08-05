package flow

// Pipe composes steps into one Step whose stages stream per item: item i
// enters the next stage as soon as its previous result lands — no barrier
// unless a stage declares one. Each stage keeps its own Policy (workers,
// admission, retry, failure mode) and its own entry in the report. An item
// quarantined or skipped by a stage bypasses every later stage and surfaces
// in the final results at its original position.
//
// Pipelines beyond arity four should prefer nesting (the stage lists
// flatten, so nesting costs nothing at runtime) over new arities.

// Pipe2 composes two steps.
func Pipe2[A, B, C any](s1 Step[A, B], s2 Step[B, C]) Step[A, C] {
	return Step[A, C]{
		Name:   pipeName(s1.Name, s2.Name),
		stages: append(stagesOf(s1), stagesOf(s2)...),
	}
}

// Pipe3 composes three steps.
func Pipe3[A, B, C, D any](s1 Step[A, B], s2 Step[B, C], s3 Step[C, D]) Step[A, D] {
	return Step[A, D]{
		Name:   pipeName(s1.Name, s2.Name, s3.Name),
		stages: append(append(stagesOf(s1), stagesOf(s2)...), stagesOf(s3)...),
	}
}

// Pipe4 composes four steps.
func Pipe4[A, B, C, D, E any](s1 Step[A, B], s2 Step[B, C], s3 Step[C, D], s4 Step[D, E]) Step[A, E] {
	return Step[A, E]{
		Name: pipeName(s1.Name, s2.Name, s3.Name, s4.Name),
		stages: append(
			append(append(stagesOf(s1), stagesOf(s2)...), stagesOf(s3)...),
			stagesOf(s4)...,
		),
	}
}

func pipeName(names ...string) string {
	joined := ""
	for index, name := range names {
		if index > 0 {
			joined += "|"
		}
		joined += name
	}
	return joined
}
