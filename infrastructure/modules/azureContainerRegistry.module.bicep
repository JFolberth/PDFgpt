@description('Name for the Form Recognizer resource.')
param acrName string
@description('SKU for Form Recognizer')
@allowed(['Basic', 'Standard', 'Premium'])
param acrSKU string = 'Standard'
@description('Location for resource.')
param location string
@description('What language was used to deploy this resource')
param language string

param keyVaultName string

resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' existing = {
  name: keyVaultName
}




resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: toLower('acr${acrName}')
  location: location
  sku: {
    name: acrSKU
  }
  properties: {
    adminUserEnabled: true
  }
  tags: {
    language: language
  }
  identity: {
    type: 'SystemAssigned'
  }
}

resource acrUserName 'Microsoft.KeyVault/vaults/secrets@2023-02-01'= {
  parent : keyVault
  name: 'acr-username'
  properties: {
    value:  acr.listCredentials().username
  }
}
resource acrPassword 'Microsoft.KeyVault/vaults/secrets@2023-02-01'= {
  parent : keyVault
  name: 'acr-password'
  properties: {
    value:  acr.listCredentials().passwords[0].value
  }
}

output acrNameOutput string = acr.name
output acrLoginServerOutput string = acr.properties.loginServer
