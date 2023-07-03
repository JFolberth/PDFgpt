@description('Name for the Form Recognizer resource.')
param formRecognizerName string
@description('SKU for Form Recognizer')
@allowed(['F0'
          'S0'])
param formRecognizerSKU string = 'S0'
@description('Location for resource.')
param location string
@description('What language was used to deploy this resource')
param language string
@description('Name of KeyVault to store values in')
param keyVaultName string

resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' existing = {
  name: keyVaultName
}


resource formRecognizer 'Microsoft.CognitiveServices/accounts@2022-12-01'={
  name: toLower('fr-${formRecognizerName}')
  location: location
  kind: 'FormRecognizer'
  sku: {
    name: formRecognizerSKU
  }
  properties: {
    customSubDomainName: formRecognizerName
  }
  identity: {
    type: 'SystemAssigned'
  }
  tags: {
    language: language
  }

}

resource formRecognizerEndpoint 'Microsoft.KeyVault/vaults/secrets@2023-02-01'= {
  parent : keyVault
  name: 'form-recognizer-endpoint'
  properties: {
    value: formRecognizer.properties.endpoint
  }
}
resource formRecognizerKey 'Microsoft.KeyVault/vaults/secrets@2023-02-01'= {
  parent : keyVault
  name: 'form-recognizer-key'
  properties: {
    value: formRecognizer.listKeys().key1
  }
}

