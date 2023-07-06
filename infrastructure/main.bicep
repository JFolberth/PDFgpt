@description('Location for all resources.')
param location string
@description('Base name that will appear for all resources.') 
param baseName string = 'pdfgptdemo'
@description('Three letter environment abreviation to denote environment that will appear in all resource names') 
param environmentName string = 'dev'
@description('Form Recognizer Sku')
param formRecognizerSKU string


targetScope = 'subscription'

var regionReference = {
  centralus: 'cus'
  eastus: 'eus'
  westus: 'wus'
  westus2: 'wus2'
}
var nameSuffix = toLower('${baseName}-${environmentName}-${regionReference[location]}')
var nameShort = toLower('${baseName}${environmentName}${regionReference[location]}')
var language = 'Bicep'
var dnsLabel = 'pdfgptdemo'
var aciImageNameTag = 'pdfgptdemo:latest'
var aciImage = 'streamlitapp'

/* Since we are mismatching scopes with a deployment at subscription and resource at Resource Group
 the main.bicep requires a resource Group deployed at the subscription scope, all modules will be at the Resource Group scope
 */
module userIdentity 'modules/userAssignedIdentity.module.bicep' = {
  name: 'userIdentityModule'
  scope: resourceGroup
  params:{
    location: location
    userIdentityName: nameSuffix
    language: language
  }
}
resource resourceGroup 'Microsoft.Resources/resourceGroups@2021-04-01' ={
  name: toLower('rg-${nameSuffix}')
  location: location
  tags:{
    Customer: 'FlavorsIaC'
    Language: language
  }
}

module formRecognizer 'modules/formRecognizer.module.bicep' ={
  name: 'formRecognizerModule'
  scope: resourceGroup
  params:{
    location: location
    language: language
    formRecognizerSKU: formRecognizerSKU
    formRecognizerName: nameSuffix
    keyVaultName: keyVault.outputs.keyVaultNameOutput
  }
}

module acr 'modules/azureContainerRegistry.module.bicep' ={
  name: 'acrModule'
  scope: resourceGroup
  params:{
    location: location
    acrName: nameShort
    language: language
    keyVaultName: keyVault.outputs.keyVaultNameOutput
  }
}

resource keyVaultValues 'Microsoft.KeyVault/vaults@2023-02-01' existing = {
  scope: resourceGroup
  name: keyVault.outputs.keyVaultNameOutput
}

module aci 'modules/azureContainerInstance.module.bicep' ={
  name: 'aciModule'
  scope: resourceGroup
  params:{
    location: location
    aciName: nameSuffix
    language: language
    acrName: acr.outputs.acrNameOutput
    dnsLabel: dnsLabel
    aciImageNameTag:'${acr.outputs.acrLoginServerOutput}/${aciImageNameTag}'
    aciImage: aciImage
    uidName: userIdentity.outputs.userIdentityNameOutput
    keyVaultName: keyVaultValues.name
    acrUserName: keyVaultValues.getSecret('acr-username')
    acrAdminPassword: keyVaultValues.getSecret('acr-password')
  }
}

module openAI 'modules/openAI.module.bicep'={
  name: 'openAIModule'
  scope: resourceGroup
  params:{
    location: location
    openAIName: nameSuffix
    language: language
    keyVaultName: keyVault.outputs.keyVaultNameOutput
  }
}

module keyVault 'modules/azureKeyVault.module.bicep'={
  name: 'keyVaultModule'
  scope: resourceGroup
  params:{
    location: location
    keyVaultName: nameSuffix
    language: language
  }
}

module storageAccount 'modules/storageAccount.module.bicep'={
  name: 'storageAccountModule'
  scope: resourceGroup
  params:{
    location: location
    storageAccountName: nameSuffix
    language: language
    uidName: userIdentity.outputs.userIdentityNameOutput
    keyVaultName:keyVault.outputs.keyVaultNameOutput
  }
}



