package gochunk

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

type Policy struct {
	MaximumFileBytes     int64 `json:"maximum_file_bytes"`
	MaximumTestdataBytes int64 `json:"maximum_testdata_bytes"`
	IncludeGenerated     bool  `json:"include_generated"`
	IncludeVendor        bool  `json:"include_vendor"`
}

func DefaultPolicy() Policy {
	return Policy{MaximumFileBytes: 1 << 20, MaximumTestdataBytes: 256 << 10}
}
func (policy Policy) Validate() error {
	if policy.MaximumFileBytes < 1 || policy.MaximumTestdataBytes < 1 {
		return errors.New("maximum Go file and testdata sizes must be positive")
	}
	return nil
}

type Admission struct {
	Repository string `json:"repository"`
	Commit     string `json:"commit"`
	Path       string `json:"path"`
	Admitted   bool   `json:"admitted"`
	Reason     string `json:"reason"`
	SizeBytes  int    `json:"size_bytes"`
	Digest     string `json:"digest,omitempty"`
}
type LoadResult struct {
	Snapshot   Snapshot       `json:"snapshot"`
	Documents  []rag.Document `json:"documents"`
	Admissions []Admission    `json:"admissions"`
}
type workspaceManifest struct {
	Path         string `json:"path"`
	Repositories []struct {
		Name         string `json:"name"`
		WorktreePath string `json:"worktreePath"`
	} `json:"repositories"`
}

func LoadCommitted(ctx context.Context, manifestPath string, policy Policy) (LoadResult, error) {
	if err := policy.Validate(); err != nil {
		return LoadResult{}, err
	}
	manifestBytes, err := os.ReadFile(manifestPath)
	if err != nil {
		return LoadResult{}, errors.Wrap(err, "read workspace manifest")
	}
	manifest := workspaceManifest{}
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		return LoadResult{}, errors.Wrap(err, "decode workspace manifest")
	}
	if len(manifest.Repositories) == 0 {
		return LoadResult{}, errors.New("workspace manifest has no repositories")
	}
	result := LoadResult{Snapshot: Snapshot{WorkspacePath: manifest.Path}}
	for _, repository := range manifest.Repositories {
		if err := ctx.Err(); err != nil {
			return LoadResult{}, err
		}
		if strings.TrimSpace(repository.Name) == "" || strings.TrimSpace(repository.WorktreePath) == "" {
			return LoadResult{}, errors.New("workspace repository requires name and worktreePath")
		}
		head, err := repositoryHead(ctx, repository.Name, repository.WorktreePath)
		if err != nil {
			return LoadResult{}, err
		}
		result.Snapshot.Repositories = append(result.Snapshot.Repositories, head)
		modulePath, err := repositoryModulePath(ctx, repository.WorktreePath)
		if err != nil {
			return LoadResult{}, errors.Wrapf(err, "read module path for %s", repository.Name)
		}
		paths, err := trackedPaths(ctx, repository.WorktreePath)
		if err != nil {
			return LoadResult{}, errors.Wrapf(err, "list tracked files for %s", repository.Name)
		}
		for _, path := range paths {
			if !strings.HasSuffix(strings.ToLower(path), ".go") {
				continue
			}
			body, err := gitOutput(ctx, repository.WorktreePath, "show", "--no-ext-diff", "HEAD:"+path)
			if err != nil {
				return LoadResult{}, errors.Wrapf(err, "read %s:%s", repository.Name, path)
			}
			admission := decideAdmission(repository.Name, head.Commit, path, body, policy)
			result.Admissions = append(result.Admissions, admission)
			if !admission.Admitted {
				continue
			}
			documentIdentity, err := digest.JSON(struct{ Repository, Commit, Path, ContentDigest string }{repository.Name, head.Commit, path, admission.Digest})
			if err != nil {
				return LoadResult{}, err
			}
			result.Documents = append(result.Documents, rag.Document{ID: "doc-" + documentIdentity[:16], SourceURI: fmt.Sprintf("git://%s@%s/%s", repository.Name, head.Commit, path), Title: path, Text: string(body), ContentDigest: admission.Digest, Metadata: map[string]string{"repository": repository.Name, "commit": head.Commit, "path": path, "package_path": packagePath(modulePath, path)}})
		}
	}
	snapshotDigest, err := digest.JSON(struct {
		Repositories []RepositoryHead `json:"repositories"`
		Policy       Policy           `json:"policy"`
		Documents    []string         `json:"documents"`
	}{result.Snapshot.Repositories, policy, documentDigests(result.Documents)})
	if err != nil {
		return LoadResult{}, err
	}
	result.Snapshot.Digest = snapshotDigest
	return result, nil
}

func repositoryHead(ctx context.Context, name, path string) (RepositoryHead, error) {
	commit, err := gitOutput(ctx, path, "rev-parse", "HEAD")
	if err != nil {
		return RepositoryHead{}, errors.Wrapf(err, "resolve %s HEAD", name)
	}
	tree, err := gitOutput(ctx, path, "rev-parse", "HEAD^{tree}")
	if err != nil {
		return RepositoryHead{}, errors.Wrapf(err, "resolve %s tree", name)
	}
	return RepositoryHead{Name: name, Path: path, Commit: strings.TrimSpace(string(commit)), Tree: strings.TrimSpace(string(tree))}, nil
}
func trackedPaths(ctx context.Context, repository string) ([]string, error) {
	output, err := gitOutput(ctx, repository, "ls-tree", "-r", "-z", "--name-only", "HEAD")
	if err != nil {
		return nil, err
	}
	parts := bytes.Split(output, []byte{0})
	paths := make([]string, 0, len(parts))
	for _, part := range parts {
		if len(part) > 0 {
			paths = append(paths, filepath.ToSlash(string(part)))
		}
	}
	sort.Strings(paths)
	return paths, nil
}
func repositoryModulePath(ctx context.Context, repository string) (string, error) {
	body, err := gitOutput(ctx, repository, "show", "--no-ext-diff", "HEAD:go.mod")
	if err != nil {
		return "", err
	}
	for _, line := range strings.Split(string(body), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 2 && fields[0] == "module" {
			return fields[1], nil
		}
	}
	return "", errors.New("go.mod has no module directive")
}
func packagePath(modulePath, path string) string {
	directory := filepath.ToSlash(filepath.Dir(path))
	if directory == "." || directory == "" {
		return modulePath
	}
	return strings.TrimSuffix(modulePath, "/") + "/" + strings.TrimPrefix(directory, "./")
}
func decideAdmission(repository, commit, path string, body []byte, policy Policy) Admission {
	admission := Admission{Repository: repository, Commit: commit, Path: path, SizeBytes: len(body)}
	switch {
	case !policy.IncludeVendor && (strings.HasPrefix(path, "vendor/") || strings.Contains(path, "/vendor/")):
		admission.Reason = "vendor"
	case int64(len(body)) > policy.MaximumFileBytes:
		admission.Reason = "file-too-large"
	case strings.Contains(path, "/testdata/") && int64(len(body)) > policy.MaximumTestdataBytes:
		admission.Reason = "testdata-too-large"
	case !policy.IncludeGenerated && IsGenerated(string(body)):
		admission.Reason = "generated"
	default:
		admission.Admitted = true
		admission.Reason = "admitted"
		admission.Digest = digest.Bytes(body)
	}
	return admission
}
func documentDigests(documents []rag.Document) []string {
	values := make([]string, len(documents))
	for index, document := range documents {
		values[index] = document.ID + ":" + document.ContentDigest
	}
	return values
}
func gitOutput(ctx context.Context, repository string, arguments ...string) ([]byte, error) {
	commandArguments := append([]string{"-C", repository}, arguments...)
	command := exec.CommandContext(ctx, "git", commandArguments...)
	output, err := command.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, errors.Errorf("git %s: %s", strings.Join(arguments, " "), strings.TrimSpace(string(exitErr.Stderr)))
		}
		return nil, errors.Wrap(err, "execute git")
	}
	return output, nil
}
