@description('Name for the Open AI resource.')
param openAIName string
@description('SKU for OpenAI')
param openAISku string = 'S0'
@description('Location for resource.')
param location string
@description('What language was used to deploy this resource')
param language string
@description('Name of KeyVault to store values in')
param keyVaultName string

resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' existing = {
  name: keyVaultName
}


resource openAI 'Microsoft.CognitiveServices/accounts@2023-10-01-preview'= {
  name: openAIName
  location: location
  sku: {
    name: openAISku
  }
  kind: 'OpenAI'
  properties: {
  }
  tags:{
    language:language
  }
}


resource gpt35Deployment 'Microsoft.CognitiveServices/accounts/deployments@2023-10-01-preview'={
  parent: openAI
  name: 'gpt-35-turbo'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-35-turbo'
      version: '0301'
      
    }
   sku: {
    capacity: 120
   }
  }
}


resource textembeddings 'Microsoft.CognitiveServices/accounts/deployments@2023-10-01-preview'={
  parent: openAI
  name: 'text-embedding-ada-002'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-ada-002'
      version: '2'
      
    }
   sku: {
    capacity: 120
   }
   raiPolicyName:'Microsoft.Default'
  }
}

resource openAIEndpoint 'Microsoft.KeyVault/vaults/secrets@2023-02-01'= {
  parent : keyVault
  name: 'openai-endpoint'
  properties: {
    value: openAI.properties.endpoint
  }
}

resource openAIKey 'Microsoft.KeyVault/vaults/secrets@2023-02-01'= {
  parent : keyVault
  name: 'openai-key'
  properties: {
    value: openAI.listKeys().key1
  }
}
