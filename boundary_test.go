package ragkit_test

import (
	"os/exec"
	"strings"
	"testing"
)

// forbiddenCoreImports are frameworks that must never leak into the ragkit
// core. Provider adapters under rag/provider/... are the only place
// LLM-framework dependencies are allowed; CLI frameworks are never allowed
// because ragkit is a library. This is the ragkit port of rag-ttc's
// TestResearchPackagesDoNotImportTheApp boundary test.
var forbiddenCoreImports = []string{
	"github.com/go-go-golems/geppetto",
	"github.com/go-go-golems/pinocchio",
	"github.com/go-go-golems/glazed",
	"github.com/spf13/cobra",
	"github.com/charmbracelet/bubbletea",
}

func TestCoreDoesNotImportAdapterFrameworks(t *testing.T) {
	const adapterPrefix = "github.com/go-go-golems/ragkit/rag/provider/"

	out, err := exec.Command("go", "list", "-f", "{{.ImportPath}}", "./...").CombinedOutput()
	if err != nil {
		t.Fatalf("go list packages: %v\n%s", err, out)
	}
	packages := strings.Fields(string(out))
	args := []string{"list", "-deps"}
	for _, packagePath := range packages {
		if !strings.HasPrefix(packagePath, adapterPrefix) {
			args = append(args, packagePath)
		}
	}

	out, err = exec.Command("go", args...).CombinedOutput()
	if err != nil {
		t.Fatalf("go list core dependencies: %v\n%s", err, out)
	}
	for _, dep := range strings.Fields(string(out)) {
		for _, forbidden := range forbiddenCoreImports {
			if strings.HasPrefix(dep, forbidden) {
				t.Errorf("core dependency tree includes forbidden import %s", dep)
			}
		}
	}
}
