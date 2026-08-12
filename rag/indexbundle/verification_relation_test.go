package indexbundle

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestVerificationRelationPersistsExactIdentitiesAcrossBatches(t *testing.T) {
	relation, err := openVerificationRelation(t.Context())
	require.NoError(t, err)
	path := relation.path
	t.Cleanup(func() { _ = relation.closeAndRemove() })

	for index := 0; index <= verificationBatchSize; index++ {
		require.NoError(t, relation.addChunk(
			t.Context(), "chunk-"+decimal(index), "doc-"+decimal(index%2), index/2, "digest-"+decimal(index),
		))
	}
	require.NoError(t, relation.finishChunks(t.Context()))

	count, err := relation.documentCount(t.Context())
	require.NoError(t, err)
	require.Equal(t, 2, count)
	digest, exists, err := relation.chunkContentDigest(t.Context(), "chunk-"+decimal(verificationBatchSize))
	require.NoError(t, err)
	require.True(t, exists)
	require.Equal(t, "digest-"+decimal(verificationBatchSize), digest)

	for index := 0; index <= verificationBatchSize; index++ {
		require.NoError(t, relation.addRepresentation(t.Context(), "rep-"+decimal(index)))
	}
	require.NoError(t, relation.finishRepresentations(t.Context()))
	require.NoError(t, relation.closeAndRemove())
	require.NoFileExists(t, path)
}

func TestVerificationRelationRejectsDuplicateIdentities(t *testing.T) {
	t.Run("chunk ID", func(t *testing.T) {
		relation, err := openVerificationRelation(t.Context())
		require.NoError(t, err)
		t.Cleanup(func() { _ = relation.closeAndRemove() })
		require.NoError(t, relation.addChunk(t.Context(), "chunk", "doc-a", 0, "digest"))
		err = relation.addChunk(t.Context(), "chunk", "doc-b", 0, "digest")
		require.ErrorContains(t, err, `duplicate chunk "chunk"`)
	})

	t.Run("document ordinal", func(t *testing.T) {
		relation, err := openVerificationRelation(t.Context())
		require.NoError(t, err)
		t.Cleanup(func() { _ = relation.closeAndRemove() })
		require.NoError(t, relation.addChunk(t.Context(), "chunk-a", "doc", 0, "digest"))
		err = relation.addChunk(t.Context(), "chunk-b", "doc", 0, "digest")
		require.ErrorContains(t, err, `duplicate ordinal 0 for document "doc"`)
	})

	t.Run("representation ID", func(t *testing.T) {
		relation, err := openVerificationRelation(t.Context())
		require.NoError(t, err)
		t.Cleanup(func() { _ = relation.closeAndRemove() })
		require.NoError(t, relation.finishChunks(t.Context()))
		require.NoError(t, relation.addRepresentation(t.Context(), "rep"))
		err = relation.addRepresentation(t.Context(), "rep")
		require.ErrorContains(t, err, `duplicate representation ID "rep"`)
	})
}

func TestVerificationRelationHonorsCancellationAndCleansUp(t *testing.T) {
	relation, err := openVerificationRelation(t.Context())
	require.NoError(t, err)
	path := relation.path

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	err = relation.addChunk(ctx, "chunk", "doc", 0, "digest")
	require.ErrorIs(t, err, context.Canceled)
	require.NoError(t, relation.closeAndRemove())
	_, err = os.Stat(path)
	require.ErrorIs(t, err, os.ErrNotExist)
}

func decimal(value int) string {
	if value == 0 {
		return "0"
	}
	var digits [20]byte
	position := len(digits)
	for value > 0 {
		position--
		digits[position] = byte('0' + value%10)
		value /= 10
	}
	return string(digits[position:])
}
