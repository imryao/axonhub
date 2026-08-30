package biz

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/internal/authz"
	"github.com/looplj/axonhub/internal/ent"
	"github.com/looplj/axonhub/internal/ent/channel"
	"github.com/looplj/axonhub/internal/ent/enttest"
	"github.com/looplj/axonhub/internal/objects"
	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/transformer/modelhub"
	"github.com/looplj/axonhub/llm/transformer/shared"
)

func TestModelHubChannelBuildsNativeResponsesOutbound(t *testing.T) {
	client := enttest.NewEntClient(t, "sqlite3", "file:ent?mode=memory&_fk=0")
	defer client.Close()

	ctx := authz.WithTestBypass(context.Background())
	entChannel := client.Channel.Create().
		SetName("ModelHub Channel").
		SetType(channel.TypeModelhub).
		SetBaseURL(modelhub.DefaultBaseURL).
		SetCredentials(objects.ChannelCredentials{APIKey: "test-ak"}).
		SetSupportedModels([]string{"gpt-5.6-sol"}).
		SetDefaultTestModel("gpt-5.6-sol").
		SaveX(ctx)

	built, err := NewChannelServiceForTest(client).buildChannelWithOutbounds(entChannel)
	require.NoError(t, err)
	require.NotNil(t, built)
	require.Equal(t, llm.APIFormatOpenAIResponse, built.Outbound.APIFormat())
	require.Same(t, built.Outbound, built.Outbounds[llm.APIFormatOpenAIResponse.String()])
	require.NotNil(t, built.Outbounds[llm.APIFormatOpenAIResponseCompact.String()])

	request, err := built.Outbound.TransformRequest(shared.WithSessionID(ctx, "session-1"), &llm.Request{
		Model:       "gpt-5.6-sol",
		RequestType: llm.RequestTypeChat,
		Messages: []llm.Message{{
			Role:    "user",
			Content: llm.MessageContent{Content: lo.ToPtr("hello")},
		}},
	})
	require.NoError(t, err)
	require.Equal(t, "test-ak", request.Query.Get(modelhub.APIKeyQueryParameter))
	require.Nil(t, request.Auth)
	require.Empty(t, request.Headers.Get("Authorization"))
	require.Equal(t, `{"session_id":"session-1"}`, request.Headers.Get("Extra"))

	var body map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(request.Body, &body))
	require.Equal(t, json.RawMessage("128000"), body["max_output_tokens"])
	require.NotEqual(t, `"hello"`, string(body["input"]))

	compactRequest, err := built.Outbounds[llm.APIFormatOpenAIResponseCompact.String()].TransformRequest(ctx, &llm.Request{
		Model:       "gpt-5.6-sol",
		RequestType: llm.RequestTypeCompact,
		Compact:     &llm.CompactRequest{},
	})
	require.NoError(t, err)
	require.Equal(t, "https://aidp.bytedance.net/api/modelhub/online/responses/compact", compactRequest.URL)
	require.Equal(t, "test-ak", compactRequest.Query.Get(modelhub.APIKeyQueryParameter))
}

func TestModelHubChannelRejectsUnsupportedCustomEndpoint(t *testing.T) {
	client := enttest.NewEntClient(t, "sqlite3", "file:ent?mode=memory&_fk=0")
	defer client.Close()

	ctx := authz.WithTestBypass(context.Background())
	entChannel := client.Channel.Create().
		SetName("ModelHub Invalid Endpoint Channel").
		SetType(channel.TypeModelhub).
		SetBaseURL(modelhub.DefaultBaseURL).
		SetCredentials(objects.ChannelCredentials{APIKey: "test-ak"}).
		SetSupportedModels([]string{"gpt-5.6-sol"}).
		SetDefaultTestModel("gpt-5.6-sol").
		SetEndpoints([]objects.ChannelEndpoint{{APIFormat: llm.APIFormatOpenAIChatCompletion.String()}}).
		SaveX(ctx)

	_, err := NewChannelServiceForTest(client).buildChannelWithOutbounds(entChannel)
	require.Error(t, err)
	require.Contains(t, err.Error(), "ModelHub supports only")
}

func TestModelHubChannelDefaultsAndValidatesBaseURL(t *testing.T) {
	client := enttest.NewEntClient(t, "sqlite3", "file:ent?mode=memory&_fk=0")
	defer client.Close()
	ctx := authz.WithTestBypass(context.Background())
	svc := NewChannelServiceForTest(client)

	created, err := svc.CreateChannel(ctx, ent.CreateChannelInput{
		Type:             channel.TypeModelhub,
		Name:             "ModelHub Default URL Channel",
		Credentials:      objects.ChannelCredentials{APIKey: "test-ak"},
		SupportedModels:  []string{"gpt-5.6-sol"},
		DefaultTestModel: "gpt-5.6-sol",
	})
	require.NoError(t, err)
	require.Equal(t, modelhub.DefaultBaseURL, created.BaseURL)

	_, err = svc.CreateChannel(ctx, ent.CreateChannelInput{
		Type:             channel.TypeModelhub,
		Name:             "ModelHub Query URL Channel",
		BaseURL:          lo.ToPtr(modelhub.DefaultBaseURL + "?ak=embedded"),
		Credentials:      objects.ChannelCredentials{APIKey: "test-ak"},
		SupportedModels:  []string{"gpt-5.6-sol"},
		DefaultTestModel: "gpt-5.6-sol",
	})
	require.Error(t, err)
	require.Contains(t, err.Error(), "invalid ModelHub base URL")
}
