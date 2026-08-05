package ragkit_test

import (
	"os/exec"
	"strings"
	"testing"
)

// forbiddenCoreImports are frameworks that must never leak into the ragkit
// core. Provider adapters (a future ragkit/providers/... tree) are the only
// place LLM-framework dependencies are allowed; CLI frameworks are never
// allowed because ragkit is a library. This is the ragkit port of rag-ttc's
// TestResearchPackagesDoNotImportTheApp boundary test.
var forbiddenCoreImports = []string{
	"github.com/go-go-golems/geppetto",
	"github.com/go-go-golems/pinocchio",
	"github.com/go-go-golems/glazed",
	"github.com/spf13/cobra",
	"github.com/charmbracelet/bubbletea",
}

func TestCoreDoesNotImportAdapterFrameworks(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "./...").CombinedOutput()
	if err != nil {
		t.Fatalf("go list -deps: %v\n%s", err, out)
	}
	deps := strings.Split(strings.TrimSpace(string(out)), "\n")
	for _, dep := range deps {
		if strings.HasPrefix(dep, "github.com/go-go-golems/ragkit/providers/") {
			continue
		}
		for _, forbidden := range forbiddenCoreImports {
			if strings.HasPrefix(dep, forbidden) {
				t.Errorf("core dependency tree includes forbidden import %s", dep)
			}
		}
	}
}
